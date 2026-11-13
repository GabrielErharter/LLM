"""
treino.py

Nucleo do Capitulo 5:
- calc_loss_batch / calc_loss_loader: cross-entropy loss
- avaliar_modelo: loss de treino e validacao (modo eval, sem gradientes)
- treinar_modelo: laco de treinamento (epocas x lotes)
- salvar_checkpoint / carregar_checkpoint: salvar e restaurar o modelo
"""
import math
import os
import time

import torch
import torch.nn.functional as F

from gpt import GPTModel
from metricas import perplexidade


# ----------------------------------------------------------------------
# Loss
# ----------------------------------------------------------------------
def calc_loss_batch(input_batch, target_batch, model, device):
    """
    Cross-entropy media sobre todas as posicoes do lote.

    logits: (B, T, V) -> (B*T, V);  alvos: (B, T) -> (B*T)
    """
    input_batch = input_batch.to(device)
    target_batch = target_batch.to(device)
    logits = model(input_batch)
    return F.cross_entropy(logits.flatten(0, 1), target_batch.flatten())


def calc_loss_loader(loader, model, device, num_batches=None):
    """Loss media sobre (ate) num_batches lotes do loader."""
    if len(loader) == 0:
        return float("nan")
    n = len(loader) if num_batches is None else min(num_batches, len(loader))
    total = 0.0
    for i, (x, y) in enumerate(loader):
        if i >= n:
            break
        total += calc_loss_batch(x, y, model, device).item()
    return total / n


def avaliar_modelo(model, train_loader, val_loader, device, eval_iter=None):
    """Loss de treino e de validacao, sem dropout e sem gradientes."""
    model.eval()
    with torch.no_grad():
        tl = calc_loss_loader(train_loader, model, device, eval_iter)
        vl = calc_loss_loader(val_loader, model, device, eval_iter)
    model.train()
    return tl, vl


# ----------------------------------------------------------------------
# Treinamento
# ----------------------------------------------------------------------
def treinar_modelo(model, train_loader, val_loader, optimizer, device,
                   num_epochs, eval_freq=20, eval_iter=5, amostra_fn=None,
                   grad_clip=None, verbose=True):
    """
    Para cada epoca, para cada lote:
        1. zera os gradientes
        2. forward + loss (cross-entropy)
        3. backward (retropropagacao)
        4. (opcional) clipping do gradiente
        5. optimizer.step() (atualiza os pesos)

    Retorna um dicionario de historico (JSON-serializavel).
    """
    hist = {"inicial": None, "passos": [], "epocas": [],
            "tempo_total_s": 0.0, "tokens_vistos": 0}

    tl, vl = avaliar_modelo(model, train_loader, val_loader, device)
    hist["inicial"] = {"train_loss": tl, "val_loss": vl}
    if verbose:
        print(f"[inicial] train={tl:.4f} (ppl {perplexidade(tl):.1f}) | "
              f"val={vl:.4f} (ppl {perplexidade(vl):.1f})")

    step, tokens_vistos = 0, 0
    t_inicio = time.perf_counter()
    model.train()

    for epoca in range(1, num_epochs + 1):
        t_ep = time.perf_counter()
        for x, y in train_loader:
            optimizer.zero_grad()
            loss = calc_loss_batch(x, y, model, device)
            loss.backward()
            if grad_clip:
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
            optimizer.step()

            step += 1
            tokens_vistos += x.numel()

            if step % eval_freq == 0:
                tl, vl = avaliar_modelo(model, train_loader, val_loader,
                                        device, eval_iter)
                hist["passos"].append({
                    "step": step, "epoca": epoca, "tokens": tokens_vistos,
                    "train_loss": tl, "val_loss": vl})
                if verbose:
                    print(f"  ep {epoca} passo {step:5d} | "
                          f"train={tl:.4f} val={vl:.4f}")

        # fim da epoca: avaliacao completa + amostra de texto
        tl, vl = avaliar_modelo(model, train_loader, val_loader, device)
        amostra = amostra_fn(model) if amostra_fn else ""
        model.train()
        hist["epocas"].append({
            "epoca": epoca, "train_loss": tl, "val_loss": vl,
            "ppl_train": perplexidade(tl), "ppl_val": perplexidade(vl),
            "tempo_s": time.perf_counter() - t_ep, "amostra": amostra})
        if verbose:
            print(f"[epoca {epoca}] train={tl:.4f} val={vl:.4f} "
                  f"(ppl val {perplexidade(vl):.1f}) | "
                  f"{time.perf_counter() - t_ep:.1f}s")

    hist["tempo_total_s"] = time.perf_counter() - t_inicio
    hist["tokens_vistos"] = tokens_vistos
    return hist


# ----------------------------------------------------------------------
# Salvar e carregar
# ----------------------------------------------------------------------
def salvar_checkpoint(caminho, model, optimizer, config, tok_info,
                      historico=None, extra=None):
    """
    Salva tudo que e necessario para retomar ou reproduzir:
    pesos (state_dict), estado do otimizador, configuracao da arquitetura,
    vocabulario do tokenizador, historico de loss e parametros do treino.
    Apenas tipos simples (tensores, numeros, textos, listas, dicts), para
    funcionar com torch.load(weights_only=True).
    """
    pasta = os.path.dirname(os.path.abspath(caminho))
    os.makedirs(pasta, exist_ok=True)
    torch.save({
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict() if optimizer else None,
        "config": config,
        "tokenizador": tok_info,
        "historico": historico,
        "extra": extra,
    }, caminho)


def carregar_checkpoint(caminho, device="cpu"):
    """Reconstroi o modelo (modo eval) a partir do checkpoint.
    Retorna (modelo, checkpoint_dict)."""
    ckpt = torch.load(caminho, map_location=device)
    model = GPTModel(ckpt["config"])
    model.load_state_dict(ckpt["model_state_dict"])
    model.to(device)
    model.eval()
    return model, ckpt
