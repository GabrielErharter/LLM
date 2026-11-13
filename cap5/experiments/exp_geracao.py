"""
exp_geracao.py - Estrategias de decodificacao (Entregavel 5).

Compara, a partir do MESMO prompt:
  greedy | amostragem com temperatura | top-k (+ temperatura)
no modelo TREINADO e num modelo NAO treinado (mesma arquitetura, pesos
aleatorios), para separar o efeito da estrategia do efeito dos pesos.

Metricas (calculadas sobre palavras da continuacao):
  distinct-1/distinct-2 : diversidade (1.0 = sem repeticao)
  repeticao-3           : fracao de trigramas repetidos (loops)
  palavras validas      : fracao de palavras que existem no texto de treino
                          (proxy de coerencia ortografica)

Uso:
    python experiments/exp_geracao.py
    python experiments/exp_geracao.py --checkpoint checkpoints/modelo_base.pt --prompt "Era uma vez"
Saida: experiments/RESULTADOS_GERACAO.md
"""
import argparse
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
RAIZ = os.path.join(AQUI, "..")
sys.path.append(os.path.join(RAIZ, "src"))

import torch  # noqa: E402

from dados import carregar_texto  # noqa: E402
from gpt import GPTModel  # noqa: E402
from metricas import (distinct_n, taxa_repeticao, taxa_palavras_validas,  # noqa: E402
                      palavras, media)
from pipeline import carregar_checkpoint, gerar_texto, escolher_device  # noqa: E402
from tokenizador import tokenizador_de_info  # noqa: E402

ESTRATEGIAS = [
    ("greedy", 0.0, None),
    ("temperatura 0.5", 0.5, None),
    ("temperatura 1.0", 1.0, None),
    ("temperatura 1.5", 1.5, None),
    ("top-k 5 (T=1.0)", 1.0, 5),
    ("top-k 20 (T=1.0)", 1.0, 20),
    ("top-k 5 (T=0.5)", 0.5, 5),
]
linhas = []


def log(t=""):
    print(t)
    linhas.append(t)


def avaliar(model, tok, cfg, prompt, novos, vocab_ref, n_seeds):
    linhas_tab, amostras = [], {}
    for nome, temp, k in ESTRATEGIAS:
        seeds = [0] if temp == 0.0 else list(range(n_seeds))  # greedy e deterministico
        d1, d2, rep, val = [], [], [], []
        for s in seeds:
            cont = gerar_texto(model, tok, prompt, novos, cfg["context_length"],
                               temp, k, seed=s, so_continuacao=True)
            if s == 0:
                amostras[nome] = cont
            ws = palavras(cont)
            d1.append(distinct_n(ws, 1))
            d2.append(distinct_n(ws, 2))
            rep.append(taxa_repeticao(ws, 3))
            val.append(taxa_palavras_validas(cont, vocab_ref))
        linhas_tab.append((nome, media(d1), media(d2), media(rep), media(val), len(seeds)))
    return linhas_tab, amostras


def imprimir(titulo, tab, amostras):
    log(f"### {titulo}\n")
    log("| estrategia | distinct-1 | distinct-2 | repeticao-3 | palavras validas | amostras |")
    log("|---|---|---|---|---|---|")
    for n, d1, d2, rep, val, ns in tab:
        log(f"| {n} | {d1:.3f} | {d2:.3f} | {rep:.3f} | {val:.3f} | {ns} |")
    log("")
    for n, txt in amostras.items():
        log(f"**{n}**\n```\n{txt[:400]}\n```\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(RAIZ, "checkpoints", "modelo_base.pt"))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--novos", type=int, default=None)
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--arquivo", default=None)
    a = ap.parse_args()

    device = escolher_device()
    treinado, ckpt = carregar_checkpoint(a.checkpoint, device)
    cfg = ckpt["config"]
    tok = tokenizador_de_info(ckpt["tokenizador"])
    extra = ckpt.get("extra") or {}
    prompt = a.prompt or extra.get("prompt")
    arquivo = a.arquivo or extra.get("arquivo")
    try:
        texto, _ = carregar_texto(arquivo)
    except FileNotFoundError:
        texto, _ = carregar_texto(None)  # usa data/texto.txt ou a amostra
    vocab_ref = set(palavras(texto))
    novos = a.novos or (300 if tok.tipo == "char" else 60)

    torch.manual_seed(123)
    inicial = GPTModel(cfg).to(device).eval()  # mesma arquitetura, nao treinado

    log("# Resultados - Estrategias de Geracao\n")
    log(f"- Checkpoint: `{os.path.basename(a.checkpoint)}` | tokenizador: {tok.tipo}")
    log(f"- Prompt (identico em todas as geracoes): `{prompt}`")
    log(f"- Tokens novos por geracao: {novos} | seeds por estrategia amostrada: {a.seeds}\n")

    tab_t, am_t = avaliar(treinado, tok, cfg, prompt, novos, vocab_ref, a.seeds)
    imprimir("Modelo TREINADO", tab_t, am_t)
    tab_i, am_i = avaliar(inicial, tok, cfg, prompt, novos, vocab_ref, a.seeds)
    imprimir("Modelo NAO TREINADO (pesos aleatorios)", tab_i, am_i)

    log("### Como separar o efeito da estrategia do efeito do treino\n")
    log("- Compare cada linha do modelo treinado com a MESMA linha do nao treinado: "
        "a diferenca vem dos pesos.")
    log("- Compare linhas diferentes dentro do MESMO modelo: a diferenca vem da estrategia.")

    saida = os.path.join(AQUI, "RESULTADOS_GERACAO.md")
    with open(saida, "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    print(f"\nSalvo: {saida}")


if __name__ == "__main__":
    main()
