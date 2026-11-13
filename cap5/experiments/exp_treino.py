"""
exp_treino.py - Experimentos de treinamento (Entregavel 4).

Varia UM parametro por vez a partir da configuracao base:
  A) taxa de aprendizado   B) tamanho do lote   C) dropout
  D) numero de epocas (lido da curva do treino base, pois nao ha
     scheduler de lr e a semente e fixa: o treino de E epocas e o prefixo
     do treino de E_max epocas)

Para cada execucao registra: configuracao, loss de treino/validacao,
perplexity, diferenca val-treino, sinal de sobreajuste, tempo, tokens/s e
uma amostra de texto (greedy, mesmo prompt).

Uso:
    python experiments/exp_treino.py              # completo
    python experiments/exp_treino.py --rapido     # teste de fumaca (2 epocas)
    python experiments/exp_treino.py --epocas 15
Saidas: experiments/RESULTADOS_TREINO.md, resultados_treino.json, figuras/
"""
import argparse
import json
import math
import os
import sys

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(AQUI, "..", "src"))

from dados import carregar_texto  # noqa: E402
from metricas import perplexidade, sinal_sobreajuste  # noqa: E402
from pipeline import (PARAMS_BASE, rodar_treino, plotar_comparacao)  # noqa: E402

FIG = os.path.join(AQUI, "figuras")
linhas = []


def log(t=""):
    print(t)
    linhas.append(t)


def resumo(nome, r):
    h = r["historico"]
    ep = h["epocas"]
    tr = [e["train_loss"] for e in ep]
    va = [e["val_loss"] for e in ep]
    sobre, melhor = sinal_sobreajuste(tr, va)
    return {
        "nome": nome, "train": tr[-1], "val": va[-1],
        "ppl_val": perplexidade(va[-1]), "gap": va[-1] - tr[-1],
        "sobreajuste": sobre, "melhor_epoca": melhor,
        "tempo": h["tempo_total_s"],
        "tok_s": h["tokens_vistos"] / h["tempo_total_s"] if h["tempo_total_s"] else float("nan"),
        "amostra": ep[-1]["amostra"], "n_params": r["n_params"],
    }


def tabela(rows, titulo_param):
    log(f"| {titulo_param} | loss treino | loss val | ppl val | val-treino | melhor epoca (val) | sobreajuste? | tempo (s) | tokens/s |")
    log("|---|---|---|---|---|---|---|---|---|")
    for s in rows:
        log(f"| {s['nome']} | {s['train']:.4f} | {s['val']:.4f} | {s['ppl_val']:.1f} | "
            f"{s['gap']:+.4f} | {s['melhor_epoca']} | {'SIM' if s['sobreajuste'] else 'nao'} | "
            f"{s['tempo']:.1f} | {s['tok_s']:.0f} |")
    log("")


def amostras(rows, prompt):
    log(f"Amostras ao final do treino (greedy, prompt `{prompt}`):\n")
    for s in rows:
        log(f"**{s['nome']}**\n```\n{s['amostra']}\n```\n")


def grupo(titulo, nome_param, chave, valores, texto, base, prompt, fmt=str):
    log("=" * 70)
    log(f"## {titulo}")
    log("=" * 70)
    resultados, rows = {}, []
    for v in valores:
        nome = f"{nome_param}={fmt(v)}"
        print(f"\n>>> {nome}")
        r = rodar_treino(texto, dict(base, **{chave: v}), prompt=prompt, verbose=False)
        resultados[nome] = r
        rows.append(resumo(nome, r))
    tabela(rows, nome_param)
    plotar_comparacao(resultados, os.path.join(FIG, f"treino_{chave}.png"), titulo)
    amostras(rows, prompt)
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquivo", default=None)
    ap.add_argument("--epocas", type=int, default=PARAMS_BASE["epocas"])
    ap.add_argument("--tokenizador", default=PARAMS_BASE["tokenizador"], choices=["char", "palavra"])
    ap.add_argument("--rapido", action="store_true")
    a = ap.parse_args()

    texto, caminho = carregar_texto(a.arquivo)
    base = dict(PARAMS_BASE, epocas=2 if a.rapido else a.epocas, tokenizador=a.tokenizador)
    prompt = texto[:40]

    log("# Resultados dos Experimentos de Treinamento\n")
    log(f"- Texto: `{os.path.basename(caminho)}` ({len(texto):,} caracteres)")
    log(f"- Configuracao base: `{json.dumps(base, ensure_ascii=False)}`")
    log(f"- Baseline (chute uniforme): perplexity = tamanho do vocabulario; loss = ln(V)")
    log("- Cada experimento altera apenas o parametro indicado.\n")

    todos = {}

    # D) epocas - usa o treino base, lido em varios pontos
    log("=" * 70)
    log("## Experimento D - Numero de epocas (curva do treino base)")
    log("=" * 70)
    print("\n>>> treino base")
    base_r = rodar_treino(texto, base, prompt=prompt, verbose=False)
    ep = base_r["historico"]["epocas"]
    ini = base_r["historico"]["inicial"]
    V = base_r["cfg"]["vocab_size"]
    log(f"Baseline ln({V}) = {math.log(V):.4f}; loss inicial: treino={ini['train_loss']:.4f}, val={ini['val_loss']:.4f}\n")
    log("| epoca | loss treino | loss val | ppl val | val-treino |")
    log("|---|---|---|---|---|")
    for e in ep:
        log(f"| {e['epoca']} | {e['train_loss']:.4f} | {e['val_loss']:.4f} | {e['ppl_val']:.1f} | {e['val_loss'] - e['train_loss']:+.4f} |")
    sob, melhor = sinal_sobreajuste([e["train_loss"] for e in ep], [e["val_loss"] for e in ep])
    log(f"\nMelhor epoca pela loss de validacao: {melhor}. Sinal de sobreajuste: {'SIM' if sob else 'nao'}.\n")
    log("Amostras (greedy, mesmo prompt) em epocas selecionadas:\n")
    for e in ep:
        if e["epoca"] in (1, max(1, len(ep) // 2), len(ep)):
            log(f"**Epoca {e['epoca']}**\n```\n{e['amostra']}\n```\n")
    plotar_comparacao({"base": base_r}, os.path.join(FIG, "treino_epocas.png"), "Curva do treino base")
    todos["base"] = resumo("base", base_r)

    if a.rapido:
        valores_lr, valores_batch, valores_drop = [1e-3], [8], [0.0]
    else:
        valores_lr, valores_batch, valores_drop = [1e-4, 5e-4, 2e-3, 1e-2], [4, 16, 64], [0.0, 0.1, 0.3]

    todos["lr"] = grupo("Experimento A - Taxa de aprendizado", "lr", "lr", valores_lr,
                        texto, base, prompt, fmt=lambda v: f"{v:g}")
    todos["batch"] = grupo("Experimento B - Tamanho do lote", "batch", "batch", valores_batch,
                           texto, base, prompt)
    todos["dropout"] = grupo("Experimento C - Dropout", "dropout", "dropout", valores_drop,
                             texto, base, prompt, fmt=lambda v: f"{v:g}")

    with open(os.path.join(AQUI, "RESULTADOS_TREINO.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))
    with open(os.path.join(AQUI, "resultados_treino.json"), "w", encoding="utf-8") as f:
        json.dump(todos, f, ensure_ascii=False, indent=2)
    print(f"\nSalvo: {os.path.join(AQUI, 'RESULTADOS_TREINO.md')}")


if __name__ == "__main__":
    main()
