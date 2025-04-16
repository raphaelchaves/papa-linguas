
import re
import spacy
nlp = spacy.load("pt_core_news_sm")

negacoes = {"não", "nunca", "jamais", "nem"}
intensificadores = {"muito": 1.5, "extremamente": 2, "super": 1.8}
atenuadores = {"um pouco": 0.5, "pouco": 0.5, "meio": 0.6}
adversativas = {"mas", "porém", "contudo", "entretanto"}

def preprocessar(texto):
    texto = texto.lower()
    texto = re.sub(r"[^\w\s]", "", texto)
    return texto.split()

# Função para aplicar Lematização
def lemmatization(tokens):
    # Convert the list of tokens back into a string
    text = " ".join(tokens)
    doc = nlp(text)  # Processando o texto com o spaCy
    lemmatized_tokens = [token.lemma_ for token in doc]  # Extraindo as lemas de cada token
    return " ".join(lemmatized_tokens)  # Return as a string

def detectar_ironia(texto):
    ironia_padroes = [
        r"só que não",
        r"🙄|😒|😂|kkk+|rsrs+",
        r"claro\.\.\.|aham", 
        r"incrível+!+"
    ]
    for padrao in ironia_padroes:
        if re.search(padrao, texto, re.IGNORECASE):
            return True
    return False
