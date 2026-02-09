import streamlit as st
import sys
from deep_translator import GoogleTranslator
import pandas as pd
from comet import download_model,load_from_checkpoint

text = st.text_input("Enter text")
translator = GoogleTranslator(source="auto", target="fr")
translated_text = translator.translate(text)
st.write(translated_text)

@st.cache_resource
def load_comet_model():
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    return load_from_checkpoint(model_path)

model = load_comet_model()
data = dict(src=text,mt=translated_text,ref=translated_text)
pred = model.predict([data])
comet_pred = pred[1]*100
out = dict(COMET_Score = comet_pred)
metrics = pd.DataFrame(out,index=[0])
st.write(metrics)








