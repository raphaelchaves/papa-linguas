
sentiment_lexicon = {
    "bom": 1, "ótimo": 2, "excelente": 3, "maravilhoso": 3,
    "ruim": -1, "horrível": -3, "péssimo": -2, "terrível": -3,
    "legal": 1, "lento": -1, "rápido": 1, "eficiente": 2, "ineficiente": -2
}

expressoes = {
    "não gostei": -2, "não recomendo": -2,
    "muito bom": 2, "amei": 2, "custo benefício": 2,
    "adorei": 3, "super recomendo": 3, "entrega rápida": 1
}

giria_lexicon = {
    "top": 2, "daora": 1, "massa": 1, "zoado": -2, "mó ruim": -2,
    "de boa": 1, "tenso": -1
}

# Unificando tudo
def get_full_lexicon():
    return {**sentiment_lexicon, **giria_lexicon, **expressoes}
