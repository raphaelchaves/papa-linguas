import os
import gdown
import zipfile
import streamlit as st
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from scipy.special import softmax
import plotly.graph_objects as go
import pandas as pd

device = torch.device("cpu")  # Forçar cpu

zip_url = "https://drive.google.com/uc?id=1dBOrHADVqiI3qeLdPwsG_yu1Mk7DdsOh"
zip_path = "/tmp/transformer_model.zip"
model_dir = "/tmp/transformer_model"

@st.cache_resource
def load_model_and_tokenizer():
    if not os.path.exists(model_dir):
        gdown.download(zip_url, zip_path, quiet=False)
        with zipfile.ZipFile(zip_path, 'r') as zip_ref:
            zip_ref.extractall(model_dir)

    tokenizer = AutoTokenizer.from_pretrained(model_dir)
    model = AutoModelForSequenceClassification.from_pretrained(
        model_dir,
        local_files_only=True,
        trust_remote_code=True,
        use_safetensors=True
    ).to(device)
    return model, tokenizer

model, tokenizer = load_model_and_tokenizer()

def highlight_sentiment(row):
    if row["Classificação"] == "Positivo":
        return ['background-color: green; color: white'] * len(row)
    elif row["Classificação"] == "Negativo":
        return ['background-color: yellow; color: black'] * len(row)
    else:
        return [''] * len(row)
        
def predict_sentiment_probability(text, model_to_use, tokenizer_to_use, device_to_use):
    model_to_use.eval()
    inputs = tokenizer_to_use(
        text,
        return_tensors="pt",
        truncation=True,
        padding=True,
        max_length=512
    )
    inputs = {key: val.to(device_to_use) for key, val in inputs.items()}

    with torch.no_grad():
        outputs = model_to_use(**inputs)

    logits = outputs.logits
    probabilities = softmax(logits.cpu().numpy(), axis=1)[0]

    prob_negativa = probabilities[0]
    prob_positiva = probabilities[1]

    return {
        "texto": text,
        "probabilidade_negativo": prob_negativa * 100,
        "probabilidade_positivo": prob_positiva * 100,
        "sentimento_predito": "Positivo" if prob_positiva > prob_negativa else "Negativo",
        "score_predito_positivo": prob_positiva
    }

# Estilo CSS
page_bg_img = """
<style>
.stApp {
    background-color: #f7f7f7;
    color: black;
}
div.stButton > button {
    color: black;
    background-color: #d1d1d1;
    padding: 10px 20px;
    border-radius: 5px;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

sidebar_style = """
<style>
[data-testid="stSidebar"] {
    background-color: #f0f2f6;
    padding: 10px;
    border-radius: 10px;
}
[data-testid="stSidebar"] * {
    color: black;
}
</style>
"""
st.markdown(sidebar_style, unsafe_allow_html=True)

# Barra lateral
st.sidebar.image(
    "https://github.com/raphaelchaves/papa-linguas/raw/main/ICMC.png",
    width=150
)
st.sidebar.title("Sobre o App")
st.sidebar.info(
    """
    Este APP foi desenvolvido para a disciplina de **PLN Linguagem Natural - SCC0633-SCC5908**.
    """
)
st.sidebar.subheader("Componentes do Grupo:")
st.sidebar.write("- Érica Ribeiro")
st.sidebar.write("- Júnior Fernandes Marques")
st.sidebar.write("- Luís Vogel")
st.sidebar.write("- Marlon José Martins")
st.sidebar.write("- Raphael Franco Chaves")
st.sidebar.write("- Thiago Ambiel")

st.sidebar.subheader("Docentes:")
st.sidebar.write("- Thiago Alexandre Salgueiro Pardo")
st.sidebar.write("- Renato Moraes Silva")

# Interface principal
st.title("Grupo Papa-Línguas")
st.write("Os Segredos da Análise de Sentimentos: Uma abordagem prática para o cálculo de probabilidades")

input_type = st.sidebar.radio("Escolha o tipo de entrada:", ("Texto", "Arquivo"))

if input_type == "Texto":
    sentence = st.text_area("Digite a frase que deseja analisar:", "")

    if st.button('Analisar Sentimento'):
        resultado = predict_sentiment_probability(sentence, model, tokenizer, device)
        sentiment = resultado["sentimento_predito"]
        label_color = "green" if sentiment == "Positivo" else "yellow"

        st.markdown(
            f"""
            <div style="
                background-color: {label_color};
                padding: 10px;
                border-radius: 10px;
                text-align: center;
                color: black;
                font-size: 18px;
                font-weight: bold;
            ">
                Resultado Final: Sentimento {sentiment}
            </div>
            """,
            unsafe_allow_html=True
        )

        labels = ["Positivo", "Negativo"]
        values = [resultado["probabilidade_positivo"], resultado["probabilidade_negativo"]]
        colors = ["green", "yellow"]

        fig = go.Figure(data=[
            go.Bar(x=labels, y=values, marker_color=colors)
        ])
        fig.update_layout(
            title="Probabilidades Estimadas",
            xaxis_title="Tipo de Sentimento",
            yaxis_title="% Probabilidade",
            template="simple_white",
            plot_bgcolor="#f7f7f7",
            paper_bgcolor="#f7f7f7"
        )

        st.plotly_chart(fig)

elif input_type == "Arquivo":
    uploaded_file = st.file_uploader("Envie um arquivo CSV ou Excel (XLSX):", type=["csv", "xlsx"])

    if uploaded_file:
        if uploaded_file.name.endswith(".csv"):
            df = pd.read_csv(uploaded_file)
        elif uploaded_file.name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file)

        st.write("Visualização do arquivo:")
        st.dataframe(df.head())

        if 'comentario' in df.columns:
            if st.button("Analisar Comentários"):
                probabilidades_positivas = []
                probabilidades_negativas = []
                classificacoes = []

                for comentario in df['comentario']:
                    comentario_str = str(comentario)
                    resultado = predict_sentiment_probability(comentario_str, model, tokenizer, device)
                    sentiment = resultado["sentimento_predito"]

                    probabilidades_positivas.append(resultado["probabilidade_positivo"] / 100)
                    probabilidades_negativas.append(resultado["probabilidade_negativo"] / 100)
                    classificacoes.append(sentiment)

                df['Probabilidade Positivo'] = probabilidades_positivas
                df['Probabilidade Negativo'] = probabilidades_negativas
                df['Classificação'] = classificacoes

                #st.write("Resultados da Análise:")
                #st.dataframe(df, use_container_width=True)
                
                st.write("Resultados da Análise:")
                styled_df = df.style.apply(highlight_sentiment, axis=1)
                st.dataframe(styled_df, use_container_width=True)

                csv = df.to_csv(index=False)
                st.download_button("Baixar Resultados (CSV)", csv, "resultados.csv", "text/csv")
        else:
            st.warning("O arquivo enviado não contém a coluna 'comentario'. Verifique e tente novamente.")
