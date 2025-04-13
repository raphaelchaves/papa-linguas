import streamlit as st
import requests
import numpy as np
from gensim.models import Word2Vec
from sklearn.linear_model import LogisticRegression
import joblib
import io

# Links dos arquivos no Google Drive
links = {
    "word2vec_model.bin": "https://drive.google.com/uc?id=1EZGEyBJdICy4tr6wsDx5X0RkLV_4-otC",
    "word2vec_model.bin.wv.vectors.npy": "https://drive.google.com/uc?id=1-WXfUuuN51J23X1J43LKB5gKPkn_IaRB",
    "word2vec_model.bin.syn1neg.npy": "https://drive.google.com/uc?id=1Vh5ST2IfvSffJkd__uswFAmqk2KSudGf",
    "logistic_regression_model.joblib": "https://drive.google.com/uc?id=1Jd-U9xITcGIHCgpfb-q0jzRk8a99S-k0"
}

# Função para baixar arquivos como bytes
def download_file_as_bytes(url):
    response = requests.get(url)
    response.raise_for_status()
    return io.BytesIO(response.content)

# Carregar o modelo Word2Vec diretamente da memória
model_bin = download_file_as_bytes(links["word2vec_model.bin"])
vectors_npy = download_file_as_bytes(links["word2vec_model.bin.wv.vectors.npy"])
syn1neg_npy = download_file_as_bytes(links["word2vec_model.bin.syn1neg.npy"])

with open("word2vec_model.bin", "wb") as f:
    f.write(model_bin.getbuffer())
with open("word2vec_model.bin.wv.vectors.npy", "wb") as f:
    f.write(vectors_npy.getbuffer())
with open("word2vec_model.bin.syn1neg.npy", "wb") as f:
    f.write(syn1neg_npy.getbuffer())

model_w2v = Word2Vec.load("word2vec_model.bin")

# Carregar o modelo Logistic Regression diretamente da memória
logistic_model_bytes = download_file_as_bytes(links["logistic_regression_model.joblib"])
logistic_regression_model = joblib.load(logistic_model_bytes)

# Função para calcular o vetor da nova frase
def preprocess_and_vectorize(sentence):
    # Tokenizar e fazer o pré-processamento (ajustar conforme necessário)
    words = sentence.split()  # Ajuste conforme seu método de tokenização
    words = [word for word in words if word in model_w2v.wv]  # Filtrar palavras que estão no modelo
    if words:
        return np.mean(model_w2v.wv[words], axis=0)
    else:
        return np.zeros(model_w2v.vector_size)

# Estilo CSS para adicionar a imagem de fundo
page_bg_img = """
<style>
.stApp {
    background-image: url("https://github.com/raphaelchaves/papa-linguas/blob/main/background_picture.jpg");
    background-size: cover;
    background-position: center;
    background-repeat: no-repeat;
    background-attachment: fixed;
}
</style>
"""
st.markdown(page_bg_img, unsafe_allow_html=True)

# Adicionar barra lateral
st.sidebar.title("Sobre o App")
st.sidebar.info(
    """
    Este app utiliza modelos de **Word2Vec** e **Regressão Logística** 
    para análise de sentimento em frases. Insira uma frase no campo ao lado e clique em **Analisar Sentimento**.
    """
)
st.sidebar.subheader("Componentes do Grupo:")
st.sidebar.write("- Raphael Franco Chaves")
st.sidebar.write("- Luís Vogel")
st.sidebar.write("- Thiago")
st.sidebar.write("- Marlon Martins")
st.sidebar.write("- Érica")
st.sidebar.write("- Júnior")

# Streamlit interface
st.title("Grupo Papas-Língua")
st.write("Digite uma frase para análise de sentimento:")

# Input de texto para o usuário
sentence = st.text_area("Digite a frase", "")

# Botão para realizar a análise
if st.button('Analisar Sentimento'):
    # Calcular o vetor para a nova frase
    sentence_vector = preprocess_and_vectorize(sentence)

    # Fazer a previsão
    y_prob = logistic_regression_model.predict_proba([sentence_vector])[0]  # Probabilidades para a nova frase
    y_pred = logistic_regression_model.predict([sentence_vector])[0]

    # Interpretação do resultado
    sentiment = "Positivo" if y_pred == 0 else "Negativo"  # Ajuste conforme suas classes

    # Exibir os resultados
    st.write(f"Sentimento: {sentiment}")
    st.write(f"Probabilidades: Positivo: {y_prob[0]:.4f}, Negativo: {y_prob[1]:.4f}")
