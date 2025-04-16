# -*- coding: utf-8 -*-
"""
Created on Tue Apr 15 19:40:45 2025

@author: rapha
"""

# https://www.inf.pucrs.br/linatural/wordpress/recursos-e-ferramentas/oplexicon/

import nltk
from nltk.tokenize import word_tokenize

nltk.download('punkt')
nltk.download('punkt_tab')

lexico = {
    "bom": 1,
    "ótimo": 2,
    "excelente": 2,
    "ruim": -1,
    "péssimo": -2,
    "horrível": -2,
    "legal": 1,
    "detestei": -2,
    "gostei": 2,
    "incrível": 2
}

# Palavras de negação
negacoes = ["não", "nunca", "jamais"]

# Intensificadores (aumentam peso em +1)
intensificadores = ["muito", "super", "extremamente"]

def analisar_sentimento(texto):
    tokens = word_tokenize(texto.lower())
    score = 0
    skip_next = False

    for i, palavra in enumerate(tokens):
        if skip_next:
            skip_next = False
            continue

        valor = lexico.get(palavra, 0)

        # Verifica negação antes da palavra
        if i > 0 and tokens[i - 1] in negacoes:
            valor *= -1

        # Verifica intensificador antes da palavra
        if i > 0 and tokens[i - 1] in intensificadores:
            valor *= 2

        score += valor

    # Classificação
    if score > 0:
        return "Positivo 😀", score
    elif score < 0:
        return "Negativo 😠", score
    else:
        return "Neutro 😐", score

# Testes
exemplos = [
    "Esse filme é ótimo!",
    "Não gostei do serviço.",
    "O produto é muito bom.",
    "A comida estava péssima.",
    "Super legal e incrível!",
    "Não é ruim, mas também não é bom."
]

for frase in exemplos:
    sentimento, score = analisar_sentimento(frase)
    print(f"Frase: \"{frase}\" → {sentimento} (score: {score})")
