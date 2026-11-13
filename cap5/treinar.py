"""
treinar.py - treina o GPT, avalia (loss/perplexity), salva o checkpoint e
prova que o modelo carregado reproduz o modelo em memoria.

Exemplos:
    python treinar.py                       # usa data/texto.txt (ou a amostra)
    python treinar.py --epocas 20 --lr 1e-3
    python treinar.py --tokenizador palavra
    python treinar.py --retomar checkpoints/modelo_base.pt --epocas 5
"""
import argparse
import json
import math
import os
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(RAIZ, "src"))

import torch  # noqa: E402

from dados import carregar_texto  # noqa: E402
from metricas import perplexidade  # noqa: E402
from pipeline import (PARAMS_BASE, rodar_treino, gerar_texto,  # noqa: E402
                      plotar_historico, salvar_checkpoint, carregar_checkpoint)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arquivo", default=None)
    ap.add_argument("--tokenizador", default=PARAMS_BASE["tokenizador"],
                    choices=["char", "palavra"])
    ap.add_argument("--epocas", type=int, default=PARAMS_BASE["epocas"])
    ap.add_argument("--batch", type=int, default=PARAMS_BASE["batch"])
    ap.add_argument("--contexto", type=int, default=PARAMS_BASE["contexto"])
    ap.add_argument("--emb", type=int, default=PARAMS_BASE["emb"])
    ap.add_argument("--heads", type=int, default=PARAMS_BASE["heads"])
    ap.add_argument("--camadas", type=int, default=PARAMS_BASE["camadas"])
    ap.add_argument("--dropout", type=float, default=PARAMS_BASE["dropout"])
    ap.add_argument("--lr", type=float, default=PARAMS_BASE["lr"])
    ap.add_argument("--wd", type=float, default=PARAMS_BASE["wd"])
    ap.add_argument("--seed", type=int, default=PARAMS_BASE["seed"])
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--saida", default=os.path.join(RAIZ, "checkpoints", "modelo_base.pt"))
    ap.add_argument("--retomar", default=None)
    a = ap.parse_args()

    texto, caminho = carregar_texto(a.arquivo)
    print(f"Texto: {caminho} ({len(texto):,} caracteres)")
    if len(texto) < 50_000:
        print("AVISO: texto pequeno; os resultados serao apenas ilustrativos.")

    p = dict(PARAMS_BASE, tokenizador=a.tokenizador, epocas=a.epocas,
             batch=a.batch, contexto=a.contexto, emb=a.emb, heads=a.heads,
             camadas=a.camadas, dropout=a.dropout, lr=a.lr, wd=a.wd,
             seed=a.seed)
    print("Parametros:", json.dumps(p, ensure_ascii=False))

    r = rodar_treino(texto, p, prompt=a.prompt, retomar=a.retomar)
    h, tok, cfg, model = r["historico"], r["tok"], r["cfg"], r["model"]
    V = cfg["vocab_size"]

    # ---- Avaliacao: estado inicial x final ----
    ini, fim = h["inicial"], h["epocas"][-1]
    print("\n=== AVALIACAO ===")
    print(f"Loss de um modelo que chuta uniformemente: ln({V}) = {math.log(V):.4f} "
          f"(perplexity = {V})")
    print(f"Inicial: train={ini['train_loss']:.4f} val={ini['val_loss']:.4f} "
          f"(ppl val {perplexidade(ini['val_loss']):.1f})")
    print(f"Final  : train={fim['train_loss']:.4f} val={fim['val_loss']:.4f} "
          f"(ppl val {fim['ppl_val']:.1f})")
    print(f"Tempo total: {h['tempo_total_s']:.1f}s | tokens vistos: {h['tokens_vistos']:,}")

    # ---- Salvar ----
    extra = {"params": p, "prompt": r["prompt"], "arquivo": caminho,
             "info_dados": r["info_dados"]}
    salvar_checkpoint(a.saida, r["model"], r["optimizer"], cfg, tok.info(),
                      historico=h, extra=extra)
    print(f"\nCheckpoint salvo em: {a.saida}")

    pasta = os.path.dirname(os.path.abspath(a.saida))
    with open(os.path.join(pasta, "historico_base.json"), "w", encoding="utf-8") as f:
        json.dump({"params": p, "n_params": r["n_params"], **h}, f,
                  ensure_ascii=False, indent=2)
    plotar_historico(h, os.path.join(pasta, "curva_loss_base.png"))

    # ---- Relatorio em Markdown ----
    linhas = ["# Resultados do treinamento base\n",
              f"- Texto: `{os.path.basename(caminho)}` ({len(texto):,} caracteres)",
              f"- Dados: {r['info_dados']}",
              f"- Parametros do treino: `{json.dumps(p, ensure_ascii=False)}`",
              f"- Parametros do modelo: {r['n_params']:,}",
              f"- Dispositivo: {r['device']}",
              f"- Tempo total: {h['tempo_total_s']:.1f}s\n",
              f"Baseline: ln({V}) = {math.log(V):.4f} (perplexity {V}).\n",
              "| epoca | loss treino | loss val | ppl treino | ppl val | tempo (s) |",
              "|---|---|---|---|---|---|",
              f"| 0 (inicial) | {ini['train_loss']:.4f} | {ini['val_loss']:.4f} | "
              f"{perplexidade(ini['train_loss']):.1f} | {perplexidade(ini['val_loss']):.1f} | - |"]
    for e in h["epocas"]:
        linhas.append(f"| {e['epoca']} | {e['train_loss']:.4f} | {e['val_loss']:.4f} | "
                      f"{e['ppl_train']:.1f} | {e['ppl_val']:.1f} | {e['tempo_s']:.1f} |")
    linhas.append("\n## Amostras (greedy, mesmo prompt) por epoca\n")
    linhas.append(f"Prompt: `{r['prompt']}`\n")
    for e in h["epocas"]:
        linhas.append(f"**Epoca {e['epoca']}**\n```\n{e['amostra']}\n```\n")
    with open(os.path.join(pasta, "RESULTADOS_TREINO_BASE.md"), "w", encoding="utf-8") as f:
        f.write("\n".join(linhas))

    # ---- Prova do salvamento/carregamento ----
    device = r["device"]
    modelo2, ckpt = carregar_checkpoint(a.saida, device)
    model.eval()
    x = torch.randint(0, V, (2, min(16, cfg["context_length"])), device=device)
    with torch.no_grad():
        iguais = torch.allclose(model(x), modelo2(x), atol=1e-6)
    print(f"Modelo carregado produz os mesmos logits: {iguais}")
    txt = gerar_texto(modelo2, tok, r["prompt"], 100, cfg["context_length"])
    print("\nTexto gerado pelo modelo CARREGADO (greedy):\n" + txt)


if __name__ == "__main__":
    main()
