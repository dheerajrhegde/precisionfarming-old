Purpose of this project is to build a precision engine that can help farmers automate their farming operations. And do so in a very optimal way. 

<img width="612" alt="image" src="https://github.com/user-attachments/assets/30268dfd-f121-43b4-b319-2b885098d775">


The visionn for the end result would be
- recommend the best time to plant the crops taking into account the climate requirements and the forecast of the best time to sell the crop
- Once planted, automate the aggregation of data
  - soil information from gauges in the farm land
  - weather information from weather APIs
  - insect and disease images from drones
- With real time data, make decisions on actions to take, define specific actions for the centralling controlled farm equipment

Current state of the project has built a RAG based core Precision Farming engine that 
- takes user inputs (did not have access to soil meters and drones),
- analyzes the specific of the crop and decides on the action plan
- gives the action plan and rationale as a narrative back to the farmer (did not have access to farm equipment)

<img width="612" alt="image" src="https://github.com/user-attachments/assets/bcb426a8-84d4-4d69-9d28-867b07f63714">

## Details of the core engine
