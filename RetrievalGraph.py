from typing import List
from typing_extensions import TypedDict
import numpy as np

from langchain import hub
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.pydantic_v1 import BaseModel, Field
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.schema import Document
from langchain_community.tools.tavily_search import TavilySearchResults

from langgraph.graph import END, START, StateGraph
from guardrails import Guard
from trulens.instrument.langchain import WithFeedbackFilterDocuments
from trulens.core import Feedback
from trulens.providers.openai import OpenAI
from trulens.instrument.langchain import TruChain

class GradeDocuments(BaseModel):
    """Binary score for relevance check on retrieved documents."""
    binary_score: str = Field(
        description="Documents are relevant to the question, 'yes' or 'no'"
    )


class GraphState(TypedDict):
    """
    Represents the state of our graph.

    Attributes:
        question: question
        generation: LLM generation
        web_search: whether to add search
        documents: list of documents
    """
    question: str
    generation: str
    web_search: str
    documents: List[str]
    groundedness: str


class RetrievalGraph:
    guard = Guard()
    guard.name = 'ChatBotGuard'
    #guard.use_many(ProfanityFree(), ToxicLanguage())

    def __init__(self):
        # Initialize Tavily
        self.web_search_tool = TavilySearchResults(k=3)

        self.llm = ChatOpenAI(model_name="gpt-4o", temperature=0)

        # Get access to Chroma vector store that has NC state agriculture information
        vectorstore = Chroma(
            persist_directory="./chroma_langchain_db",
            collection_name="agriculture",
            embedding_function=OpenAIEmbeddings())
        self.retriever = vectorstore.as_retriever()

        # RAG Chain for checking relevance of retrieved documents
        prompt = hub.pull("rlm/rag-prompt")
        self.rag_chain = prompt | self.llm | StrOutputParser()
        system = """You are a grader assessing relevance of a retrieved document to a user question. \n 
            If the document contains keyword(s) or semantic meaning related to the question, grade it as relevant. \n
            Give a binary score 'yes' or 'no' score to indicate whether the document is relevant to the question."""
        self.grade_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                ("human", "Retrieved document: \n\n {document} \n\n User question: {question}"),
            ]
        )
        self.structured_llm_grader = self.llm.with_structured_output(GradeDocuments)
        #self.retrieval_grader = grade_prompt | structured_llm_grader

        # Add Guardrails to grade_documents LLM call
        """self.retrieval_grader = self.guard(
            self.grade_prompt | self.structured_llm_grader,
            schema={
                "binary_score": {
                    "type": "string",
                    "enum": ["yes", "no"],
                    "description": "Binary score for relevance of documents",
                }
            },
        )"""

        # Prompt
        system = """You a question re-writer that converts an input question to a better version that is optimized \n 
             for web search. Look at the input and try to reason about the underlying semantic intent / meaning."""
        re_write_prompt = ChatPromptTemplate.from_messages(
            [
                ("system", system),
                (
                    "human",
                    "Here is the initial question: \n\n {question} \n Formulate an improved question.",
                ),
            ]
        )

        self.question_rewriter = re_write_prompt | self.llm | StrOutputParser()

        workflow = StateGraph(GraphState)

        # Define the nodes
        workflow.add_node("retrieve", self.retrieve)  # retrieve with content relevance score
        workflow.add_node("generate", self.generate)  # generate
        workflow.add_node("transform_query", self.transform_query)  # transform_query
        workflow.add_node("web_search_node", self.web_search)  # web search

        # Build graph
        workflow.add_edge(START, "retrieve")

        workflow.add_conditional_edges(
            "retrieve",
            self.nothing_retrieved,
            {
                "web_search": "web_search_node",
                "generate": "generate",
            },
        )

        workflow.add_edge("web_search_node", "generate")

        workflow.add_conditional_edges(
            "generate",
            self.not_grounded,
            {
                "notGrounded": "transform_query",
                "grounded": END
            }
        )

        # Compile
        self.app = workflow.compile()

    def invoke(self, question):
        return self.app.invoke({"question": question})["generation"]


    def retrieve(self, state):
        provider = OpenAI()
        f_context_relevance_score = Feedback(provider.context_relevance)
        tru_recorder = TruChain(self.llm, app_name="RetrievalGraph", app_version="Chain1",
                                feedbacks=[f_context_relevance_score])
        filtered_retriever = WithFeedbackFilterDocuments.of_retriever(
            retriever=self.retriever, feedback=f_context_relevance_score, threshold=0.75
        )
        question = state["question"]
        # Retrieval
        documents = filtered_retriever.get_relevant_documents(question)
        #print(documents)
        return {"documents": documents, "question": question}

    def generate(self, state):
        question = state["question"]
        documents = state["documents"]
        provider = OpenAI()
        generation = self.rag_chain.invoke({"context": documents, "question": question})

        from langchain_upstage import UpstageGroundednessCheck
        import os
        groundedness_check = UpstageGroundednessCheck()

        request_input = {
            "context": documents,
            "answer": generation,
        }

        response = groundedness_check.invoke(request_input)
        print("Groundedness response: ", response)
        return {"documents": documents, "question": question, "generation": generation, "groundedness": response}


    def transform_query(self, state):
        """
        Transform the query to produce a better question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates question key with a re-phrased question
        """

        question = state["question"]
        documents = state["documents"]

        # Re-write question
        better_question = self.question_rewriter.invoke({"question": question})
        return {"documents": documents, "question": better_question}

    def web_search(self, state):
        """
        Web search based on the re-phrased question.

        Args:
            state (dict): The current graph state

        Returns:
            state (dict): Updates documents key with appended web results
        """

        question = state["question"]
        documents = state["documents"]

        # Web search
        docs = self.web_search_tool.invoke({"query": question})
        web_results = "\n".join([d["content"] for d in docs])
        web_results = Document(page_content=web_results)
        documents.append(web_results)

        return {"documents": documents, "question": question}

    def nothing_retrieved(self, state):
        documents = state["documents"]
        if len(documents) == 0:
            return "web_search"
        else:
            return "generate"

    def not_grounded(self, state):
        return state["groundedness"]




if __name__ == "__main__":
    graph = RetrievalGraph()
    state = graph.invoke("What is the best way to grow corn?")
    print(state)