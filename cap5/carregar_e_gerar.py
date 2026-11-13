"""
carregar_e_gerar.py - carrega um checkpoint e gera texto (sem retreinar).

    python carregar_e_gerar.py --prompt "Era uma vez" --novos 200
    python carregar_e_gerar.py --temperatura 0.8 --top-k 20 --seed 1
"""
import argparse
import os
import sys

RAIZ = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(RAIZ, "src"))

from pipeline import carregar_checkpoint, gerar_texto, escolher_device  # noqa: E402
from tokenizador import tokenizador_de_info  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.path.join(RAIZ, "checkpoints", "modelo_base.pt"))
    ap.add_argument("--prompt", default=None)
    ap.add_argument("--novos", type=int, default=200)
    ap.add_argument("--temperatura", type=float, default=0.0)
    ap.add_argument("--top-k", type=int, default=None, dest="top_k")
    ap.add_argument("--seed", type=int, default=123)
    a = ap.parse_args()

    device = escolher_device()
    model, ckpt = carregar_checkpoint(a.checkpoint, device)
    tok = tokenizador_de_info(ckpt["tokenizador"])
    cfg = ckpt["config"]
    prompt = a.prompt or (ckpt.get("extra") or {}).get("prompt") or tok.decode([0])

    print("Configuracao do modelo:", cfg)
    print(f"Tokenizador: {tok.tipo} (vocab {len(tok)})")
    print(f"Prompt: {prompt!r} | temperatura={a.temperatura} top_k={a.top_k}\n")
    print(gerar_texto(model, tok, prompt, a.novos, cfg["context_length"],
                      a.temperatura, a.top_k, seed=a.seed))


if __name__ == "__main__":
    main()
