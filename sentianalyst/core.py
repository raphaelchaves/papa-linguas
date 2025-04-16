
import re
from .utils import preprocessar, detectar_ironia, lemmatization, negacoes, intensificadores, atenuadores, adversativas

def analisar_frase(frase, lexicon):
    tokens = preprocessar(frase)
    # tokens = lemmatization(tokens)
    score = 0
    i = 0
    negando = False
    modificador = 1.0
    palavras_sentimentais = 0

    while i < len(tokens):
        palavra = tokens[i]
        bigrama = f"{palavra} {tokens[i+1]}" if i+1 < len(tokens) else None

        if palavra in adversativas:
            score = 0

        elif palavra in negacoes:
            negando = True

        elif palavra in intensificadores:
            modificador = intensificadores[palavra]

        elif palavra in atenuadores:
            modificador = atenuadores[palavra]

        elif bigrama and bigrama in lexicon:
            valor = lexicon[bigrama]
            if negando:
                valor *= -1
                negando = False
            score += valor * modificador
            palavras_sentimentais += 1
            modificador = 1.0
            i += 1

        elif palavra in lexicon:
            valor = lexicon[palavra]
            if negando:
                valor *= -1
                negando = False
            score += valor * modificador
            palavras_sentimentais += 1
            modificador = 1.0

        i += 1

    return score, palavras_sentimentais

def analisar_sentimento(texto, lexicon):
    frases = re.split(r'[.!?]', texto)
    total_score = 0
    total_sentimentos = 0

    for i, frase in enumerate(frases):
        if not frase.strip():
            continue
        peso = 1.5 if i == len(frases) - 2 else 1.0
        score_frase, count = analisar_frase(frase, lexicon)
        total_score += score_frase * peso
        total_sentimentos += count

    if detectar_ironia(texto):
        total_score *= -1.2

    if total_sentimentos > 0:
        total_score /= total_sentimentos

    return total_score

def classificar(score, limiar=0.0):
    return "positivo" if score > limiar else "negativo"

