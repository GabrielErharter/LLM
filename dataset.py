"""
dataset.py

Implementa a estrutura de Dataset/DataLoader do PyTorch para organizar
os dados em lotes (batches), prontos para o treinamento do modelo.

Segue o padrão apresentado no Capítulo 2: GPTDatasetV1 + DataLoader.
Execute na máquina local, onde o PyTorch já está instalado.
"""

import torch
from torch.utils.data import Dataset, DataLoader


class GPTDataset(Dataset):
    """
    Dataset que aplica a janela deslizante sobre uma sequência de
    Token IDs, produzindo pares (input_ids, target_ids).
    """

    def __init__(self, token_ids: list[int], context_size: int, stride: int):
        self.input_ids = []
        self.target_ids = []

        for i in range(0, len(token_ids) - context_size, stride):
            input_chunk = token_ids[i: i + context_size]
            target_chunk = token_ids[i + 1: i + context_size + 1]
            self.input_ids.append(torch.tensor(input_chunk))
            self.target_ids.append(torch.tensor(target_chunk))

    def __len__(self):
        return len(self.input_ids)

    def __getitem__(self, idx):
        return self.input_ids[idx], self.target_ids[idx]


def create_dataloader(
    token_ids: list[int],
    batch_size: int = 4,
    context_size: int = 4,
    stride: int = 4,
    shuffle: bool = True,
    drop_last: bool = True,
) -> DataLoader:
    """
    Cria um DataLoader pronto para uso, encapsulando o GPTDataset.

    Args:
        token_ids: lista de Token IDs do texto completo.
        batch_size: quantidade de amostras por lote.
        context_size: tamanho da janela de contexto.
        stride: passo do deslizamento da janela.
        shuffle: embaralha as amostras a cada época.
        drop_last: descarta o último lote se for menor que batch_size.
    """
    dataset = GPTDataset(token_ids, context_size, stride)
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        drop_last=drop_last,
    )
    return dataloader


if __name__ == "__main__":
    # ---- Demonstração ----
    token_ids_ficticios = list(range(1, 41))  # 40 tokens de exemplo

    dataloader = create_dataloader(
        token_ids_ficticios,
        batch_size=4,
        context_size=4,
        stride=4,
        shuffle=False,
    )

    print(f"Quantidade de lotes gerados: {len(dataloader)}\n")

    for batch_idx, (inputs, targets) in enumerate(dataloader):
        print(f"Lote {batch_idx}")
        print(f"  Entrada (shape={tuple(inputs.shape)}):\n{inputs}")
        print(f"  Alvo    (shape={tuple(targets.shape)}):\n{targets}")
        print("-" * 50)
