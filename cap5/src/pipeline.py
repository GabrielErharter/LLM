"""
pipeline.py

Funcoes de alto nivel reutilizadas pelos scripts e experimentos:
preparar dados, montar o modelo, treinar, gerar texto e plotar curvas.
"""
import os
import random

import torch

from dados import dividir_treino_validacao, criar_dataloader
from gpt import GPTModel, generate, count_parameters
from tokenizador import criar_tokenizador, tokenizador_de_info
from treino import treinar_modelo, salvar_checkpoint, carregar_checkpoint  # noqa: F401
from metricas import perplexidade

PARAMS_BASE = {
    "tokenizador": "char",   # "char" ou "palavra"
    "contexto": 64,
    "batch": 16,
    "emb": 128,
    "heads": 4,
    "camadas": 4,
    "dropout": 0.1,
    "lr": 5e-4,
    "wd": 0.1,
    "epocas": 10,
    "split": 0.9,
    "seed": 123,
    "stride": None,          # None -> igual ao contexto (sem sobreposicao)
    "eval_freq": 20,
    "eval_iter": 5,
    "grad_clip": None,
}


def escolher_device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")


def montar_config(vocab_size, p):
    assert p["emb"] % p["heads"] == 0, "emb precisa ser divisivel por heads"
    return {
        "vocab_size": vocab_size,
        "context_length": p["contexto"],
        "emb_dim": p["emb"],
        "n_heads": p["heads"],
        "n_layers": p["camadas"],
        "drop_rate": p["dropout"],
        "qkv_bias": False,
        "ff_dim": None,
    }


def preparar(texto, p, tok=None):
    """Tokeniza, divide em treino/validacao e cria os DataLoaders."""
    if tok is None:
        tok = criar_tokenizador(p["tokenizador"], texto)  # vocab do texto todo
    ids = tok.encode(texto)
    tr, va = dividir_treino_validacao(ids, p["split"])
    stride = p["stride"] or p["contexto"]
    train_loader = criar_dataloader(tr, p["batch"], p["contexto"], stride,
                                    shuffle=True, drop_last=True)
    val_loader = criar_dataloader(va, p["batch"], p["contexto"], stride,
                                  shuffle=False, drop_last=False)
    if len(train_loader) == 0:
        raise ValueError(
            f"Nenhum lote de treino: so ha {len(tr)} tokens de treino, o que nao "
            f"preenche um lote de {p['batch']} janelas de {p['contexto']} tokens. "
            f"Reduza --batch/--contexto ou use um texto maior.")
    info = {"n_tokens": len(ids), "n_treino": len(tr), "n_validacao": len(va),
            "lotes_treino": len(train_loader), "lotes_validacao": len(val_loader),
            "vocab": len(tok)}
    return tok, train_loader, val_loader, info


@torch.no_grad()
def gerar_texto(model, tok, prompt, max_novos, contexto, temperatura=0.0,
                top_k=None, seed=None, so_continuacao=False):
    """Gera texto a partir de um prompt (devolve prompt + continuacao,
    ou so a continuacao)."""
    ids = tok.encode(prompt)
    if not ids:
        raise ValueError("Prompt vazio apos a tokenizacao.")
    device = next(model.parameters()).device
    estava_treinando = model.training
    if seed is not None:
        torch.manual_seed(seed)
    x = torch.tensor([ids], device=device)
    out = generate(model, x, max_novos, contexto,
                   temperature=temperatura, top_k=top_k)
    if estava_treinando:
        model.train()
    novos = out[0, len(ids):] if so_continuacao else out[0]
    return tok.decode(novos.tolist())


def prompt_padrao(texto, n_chars=40):
    return texto[:n_chars]


def rodar_treino(texto, p, prompt=None, retomar=None, verbose=True):
    """Treina um modelo com os parametros p e devolve um dicionario de
    resultados (modelo, tokenizador, historico, tempo, etc.)."""
    device = escolher_device()
    random.seed(p["seed"])
    torch.manual_seed(p["seed"])

    if retomar:
        ckpt = torch.load(retomar, map_location=device)
        cfg = ckpt["config"]
        tok = tokenizador_de_info(ckpt["tokenizador"])
        p = dict(p, contexto=cfg["context_length"], tokenizador=tok.tipo)
    else:
        ckpt, cfg, tok = None, None, None

    tok, train_loader, val_loader, info = preparar(texto, p, tok)
    if cfg is None:
        cfg = montar_config(len(tok), p)

    model = GPTModel(cfg).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=p["lr"],
                                  weight_decay=p["wd"])
    if ckpt is not None:
        model.load_state_dict(ckpt["model_state_dict"])
        if ckpt.get("optimizer_state_dict"):
            optimizer.load_state_dict(ckpt["optimizer_state_dict"])

    prompt = prompt or prompt_padrao(texto)
    n_amostra = 150 if tok.tipo == "char" else 40

    def amostra_fn(m):
        return gerar_texto(m, tok, prompt, n_amostra, cfg["context_length"],
                           temperatura=0.0)

    if verbose:
        print(f"Dispositivo: {device} | parametros: {count_parameters(model):,}")
        print(f"Dados: {info}")
    hist = treinar_modelo(model, train_loader, val_loader, optimizer, device,
                          p["epocas"], p["eval_freq"], p["eval_iter"],
                          amostra_fn, p["grad_clip"], verbose)
    return {"params": p, "cfg": cfg, "n_params": count_parameters(model),
            "info_dados": info, "historico": hist, "model": model,
            "tok": tok, "optimizer": optimizer, "prompt": prompt,
            "device": str(device)}


# ----------------------------------------------------------------------
# Graficos
# ----------------------------------------------------------------------
def plotar_historico(hist, caminho, titulo="Loss de treino e validacao"):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("AVISO: matplotlib nao instalado; grafico nao gerado (pip install matplotlib).")
        return

    os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
    passos = hist["passos"]
    x = [0] + [s["step"] for s in passos]
    tr = [hist["inicial"]["train_loss"]] + [s["train_loss"] for s in passos]
    va = [hist["inicial"]["val_loss"]] + [s["val_loss"] for s in passos]
    plt.figure(figsize=(7, 4))
    plt.plot(x, tr, label="treino")
    plt.plot(x, va, "--", label="validacao")
    plt.xlabel("passo (lote)")
    plt.ylabel("cross-entropy loss")
    plt.title(titulo)
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()


def plotar_comparacao(resultados, caminho, titulo):
    """Curvas de loss de validacao (e treino pontilhado) por epoca."""
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("AVISO: matplotlib nao instalado; grafico nao gerado (pip install matplotlib).")
        return

    os.makedirs(os.path.dirname(os.path.abspath(caminho)), exist_ok=True)
    plt.figure(figsize=(7, 4.5))
    for nome, r in resultados.items():
        ep = r["historico"]["epocas"]
        xs = [0] + [e["epoca"] for e in ep]
        va = [r["historico"]["inicial"]["val_loss"]] + [e["val_loss"] for e in ep]
        tr = [r["historico"]["inicial"]["train_loss"]] + [e["train_loss"] for e in ep]
        linha, = plt.plot(xs, va, label=f"{nome} (val)")
        plt.plot(xs, tr, ":", color=linha.get_color(), label=f"{nome} (treino)")
    plt.xlabel("epoca")
    plt.ylabel("cross-entropy loss")
    plt.title(titulo)
    plt.legend(fontsize=7)
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(caminho, dpi=120)
    plt.close()
