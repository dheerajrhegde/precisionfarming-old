import PrecisionFarming
import streamlit as st
from keras.preprocessing import image

# Set up the page configuration
st.set_page_config(
    page_title="Precision Farming",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

st.title("Precision Farming")
if "pf" in st.session_state:
    pf = st.session_state["pf"]
else:
    pf = PrecisionFarming.PrecisionFarming()

col1, col2 = st.columns([0.3, 0.7])

with col1:
    with st.form(key="farmer data"):
        ph = st.number_input("Soil pH", value=6.5, step=0.1)
        moisture = st.number_input("Soil Moisture", value=30, step=1)
        location = st.text_input("Location", value="Concord, NC")
        area = st.number_input("Area (acres)", value=10, step=1)
        crop = st.selectbox(
            "What crop do you want to get information for?",
            ("Corn", "Soybean", "Cotton")
        )
        insect = st.file_uploader("Upload an image of an insect", type=["jpg", "png"])
        leaf = st.file_uploader("Upload an image of a leaf", type=["jpg", "png"])
        submitted = st.form_submit_button("Get Insights")

with col2:
    if submitted:
        insect_img = image.load_img(insect, target_size=(224, 224))
        leaf_img = image.load_img(leaf, target_size=(224, 224))
        st.markdown(pf.get_insights(ph, moisture, location, area, crop, insect_img, leaf_img))
    else:
        st.markdown("Please fill out the form to get insights.")

