import streamlit as st
import sys
from deep_translator import GoogleTranslator
import pandas as pd
from comet import download_model,load_from_checkpoint

text = st.text_input("Enter text")
translator = GoogleTranslator(source="auto", target="fr")
translated_text = translator.translate(text)
st.write(translated_text)


model_path = download_model("Unbabel/wmt22-comet-da")
model = load_from_checkpoint(model_path)
data = dict(src=text,mt=translated_text,ref=translated_text)
pred = model.predict([data])
comet_pred = pred[1]*100;comet_pred
out = dict(COMET_Score = comet_pred)
metrics = pd.DataFrame(out,index=[0])
st.write(metrics)








