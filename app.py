import streamlit as st
from deep_translator import GoogleTranslator
import pandas as pd

@st.cache_resource
def load_comet_model():
    from comet import download_model, load_from_checkpoint
    model_path = download_model("Unbabel/wmt22-comet-da")
    return load_from_checkpoint(model_path)

st.title("Translation + COMET Score")

text = st.text_input("Enter text")

if text:
    translator = GoogleTranslator(source="auto", target="fr")
    translated_text = translator.translate(text)
    st.write("Translated text:", translated_text)

    model = load_comet_model()

    data = [{
        "src": text,
        "mt": translated_text,
        "ref": translated_text  # ⚠️ not ideal, see note below
    }]

    prediction = model.predict(data, batch_size=1, gpus=0)

    comet_score = prediction.scores[0] * 100

    metrics = pd.DataFrame(
        {"COMET_Score": [comet_score]}
    )

    st.write(metrics)
