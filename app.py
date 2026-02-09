import streamlit as st
import sys
from deep_translator import GoogleTranslator
from pandas as pd
from comet import download_model,load_from_checkpoint

text = st.text_input("Enter text")
translator = GoogleTranslator(source="auto", target="fr")
translated_text = translator.translate(text)
st.write(translated_text)


model_path = download_model("Unbabel/wmt22-comet-da")






