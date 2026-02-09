import streamlit as st
import sys
from deep_translator import GoogleTranslator

text = st.text_input("Enter text")
translator = GoogleTranslator(source="auto", target="fr")
translated_text = translator.translate(text)
st.write(translated_text)
