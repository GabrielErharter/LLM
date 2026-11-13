"""
metricas.py (sem dependencia de PyTorch)

Metricas usadas na avaliacao do treinamento e na comparacao de
estrategias de geracao.
"""
import math
import re


def perplexidade(loss):
    """Perplexity = exp(loss). Interpretacao: 'numero efetivo de tokens
    igualmente provaveis' entre os quais o modelo hesita."""
    return math.exp(min(loss, 50.0))


def palavras(texto):
    return re.findall(r"\w+", texto.lower())


def distinct_n(itens, n):
    """Fracao de n-gramas distintos (1.0 = nenhuma repeticao)."""
    grams = [tuple(itens[i:i + n]) for i in range(len(itens) - n + 1)]
    return len(set(grams)) / len(grams) if grams else float("nan")


def taxa_repeticao(itens, n=3):
    """Fracao de n-gramas que sao repeticoes de outros (0.0 = nenhuma)."""
    d = distinct_n(itens, n)
    return 1.0 - d if d == d else float("nan")


def taxa_palavras_validas(texto, conjunto_palavras):
    """Fracao das palavras geradas que existem no texto de treino.
    Proxy simples de coerencia ortografica."""
    ws = palavras(texto)
    if not ws:
        return float("nan")
    return sum(w in conjunto_palavras for w in ws) / len(ws)


def sinal_sobreajuste(train_losses, val_losses, tolerancia=0.02):
    """
    Recebe listas de loss por epoca. Retorna (sobreajuste, melhor_epoca).
    Ha sinal de sobreajuste quando a loss de validacao atinge o minimo
    antes da ultima epoca e depois sobe mais que `tolerancia`.
    """
    if len(val_losses) < 2:
        return False, (1 if val_losses else None)
    k = min(range(len(val_losses)), key=lambda i: val_losses[i])
    sobe = (val_losses[-1] - val_losses[k]) > tolerancia
    return (sobe and k < len(val_losses) - 1), k + 1


def media(valores):
    v = [x for x in valores if x == x]
    return sum(v) / len(v) if v else float("nan")
