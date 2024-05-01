import streamlit as st

st.set_page_config(page_title="Basics",
                   page_icon=":bar_chart:",
                   layout="wide"
                   )
st.title("EDA Analysis Results")

val = st.slider("Select a value", 0, 100, 0, step=1)

st.write(val)

btn1 = st.button("Click Me")

if btn1:
    st.write("I am clicked")