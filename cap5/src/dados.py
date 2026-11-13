"""
dados.py

Carregamento do texto, divisao treino/validacao e DataLoaders.

A divisao e feita por POSICAO (primeiros X% para treino, o restante para
validacao), sem embaralhar antes: assim nenhuma janela de validacao
compartilha tokens com o treino.
"""
import os

import torch
from torch.utils.data import Dataset, DataLoader

RAIZ = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..")


def carregar_texto(arquivo=None):
    """Le o texto de treino. Sem argumento, procura data/texto.txt e, se
    nao existir, usa data/amostra.txt (apenas para teste rapido)."""
    candidatos = [arquivo] if arquivo else [
        os.path.join(RAIZ, "data", "texto.txt"),
        os.path.join(RAIZ, "data", "amostra.txt"),
    ]
    for c in candidatos:
        if c and os.path.exists(c):
            for enc in ("utf-8-sig", "latin-1"):
                try:
                    with open(c, encoding=enc) as f:
                        return f.read(), os.path.abspath(c)
                except UnicodeDecodeError:
                    continue
    raise FileNotFoundError(
        "Nenhum texto encontrado. Coloque um arquivo em data/texto.txt "
        "ou use --arquivo caminho.txt"
    )


def dividir_treino_validacao(token_ids, proporcao=0.9):
    corte = int(len(token_ids) * proporcao)
    return token_ids[:corte], token_ids[corte:]


class GPTDataset(Dataset):
    """Janela deslizante: entrada = tokens [i, i+T), alvo = [i+1, i+T+1)."""

    def __init__(self, token_ids, context_length, stride):
        if len(token_ids) <= context_length:
            raise ValueError(
                f"Texto curto demais ({len(token_ids)} tokens) para "
                f"context_length={context_length}. Use um texto maior ou "
                f"reduza o contexto."
            )
        self.inputs, self.targets = [], []
        for i in range(0, len(token_ids) - context_length, stride):
            self.inputs.append(torch.tensor(token_ids[i:i + context_length]))
            self.targets.append(torch.tensor(token_ids[i + 1:i + context_length + 1]))

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, idx):
        return self.inputs[idx], self.targets[idx]


def criar_dataloader(token_ids, batch_size, context_length, stride,
                     shuffle, drop_last):
    ds = GPTDataset(token_ids, context_length, stride)
    return DataLoader(ds, batch_size=batch_size, shuffle=shuffle,
                      drop_last=drop_last)
