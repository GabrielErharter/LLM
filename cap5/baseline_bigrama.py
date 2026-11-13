"""
baseline_bigrama.py (sem PyTorch)

Modelo estatistico de bigramas com suavizacao de Laplace, usando o MESMO
texto, tokenizador e divisao treino/validacao do GPT. Serve de regua:
um GPT que aprende de verdade deve ficar abaixo da loss de validacao deste
baseline (ou, ao menos, proximo, em textos muito pequenos).

    python baseline_bigrama.py
    python baseline_bigrama.py --arquivo data/texto.txt --tokenizador palavra
"""
import argparse
import math
import os
import sys
from collections import defaultdict

sys.path.append(os.path.join(os.path.dirname(os.path.abspath(__file__)), "src"))

from tokenizador import criar_tokenizador  # noqa: E402
from metricas import perplexidade  # noqa: E402


def carregar(arquivo):
    raiz = os.path.dirname(os.path.abspath(__file__))
    cands = [arquivo] if arquivo else [os.path.join(raiz, "data", "texto.txt"),
                                       os.path.join(raiz, "data", "amostra.txt")]
    for c in cands:
        if c and os.path.exists(c):
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    return open(c, encoding=enc).read(), c
                except UnicodeDecodeError:
                    pass
    raise FileNotFoundError("Nenhum texto encontrado.")


def loss_bigrama(ids_treino, ids_teste, V, alpha=1.0):
    cont = defaultdict(lambda: defaultdict(int))
    tot = defaultdict(int)
    for a, b in zip(ids_treino[:-1], ids_treino[1:]):
        cont[a][b] += 1
        tot[a] += 1

    def nll(ids):
        s, n = 0.0, 0
        for a, b in zip(ids[:-1], ids[1:]):
            p = (cont[a][b] + alpha) / (tot[a] + alpha * V)
            s -= math.log(p)
            n += 1
        return s / n

    return nll(ids_treino), nll(ids_teste)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquivo", default=None)
    ap.add_argument("--tokenizador", default="char", choices=["char", "palavra"])
    ap.add_argument("--split", type=float, default=0.9)
    a = ap.parse_args()

    texto, caminho = carregar(a.arquivo)
    tok = criar_tokenizador(a.tokenizador, texto)
    ids = tok.encode(texto)
    corte = int(len(ids) * a.split)
    tr, va = ids[:corte], ids[corte:]
    V = len(tok)
    ltr, lva = loss_bigrama(tr, va, V)
    print(f"Texto: {caminho} | tokenizador: {tok.tipo} | V={V} | treino={len(tr)} val={len(va)}")
    print(f"Chute uniforme : loss={math.log(V):.4f}  ppl={V}")
    print(f"Bigrama treino : loss={ltr:.4f}  ppl={perplexidade(ltr):.1f}")
    print(f"Bigrama val    : loss={lva:.4f}  ppl={perplexidade(lva):.1f}")


if __name__ == "__main__":
    main()
