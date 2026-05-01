import streamlit as st

from config import (
    APP_TITLE,
    APP_DESCRIPTION,
    EPISODE_LENGTH,
    MAX_INVENTORY,
    INITIAL_INVENTORY,
    COST,
)


st.set_page_config(page_title=APP_TITLE, layout="centered")

st.title(APP_TITLE)
st.write(APP_DESCRIPTION)

st.subheader("Default configuration")

st.write(f"Episode length: **{EPISODE_LENGTH}**")
st.write(f"Maximum inventory: **{MAX_INVENTORY}**")
st.write(f"Initial inventory: **{INITIAL_INVENTORY}**")

st.write(f"Cost: **{COST}**")

st.success("Streamlit app is running.")