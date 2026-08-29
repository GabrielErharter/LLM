"""
main.py

Integra o pipeline completo pedido na Sprint:

Texto -> Tokenizacao -> Tokens -> Token IDs -> Sequencias de treinamento
      -> Embeddings -> Positional Embeddings -> Lote de dados -> Entrada do modelo

Execute este arquivo na sua maquina local (onde o PyTorch ja esta
instalado e validado): python main.py
"""

import sys
import os

sys.path.append(os.path.join(os.path.dirname(__file__), "src"))

import torch  # noqa: E402
from tokenizer import Vocabulary, SimpleTokenizer  # noqa: E402
from dataset import create_dataloader  # noqa: E402
from embeddings import (  # noqa: E402
    build_token_embedding_layer,
    build_positional_embedding_layer,
    compute_input_embeddings,
)


def main():
    # ---- Configurações do pipeline ----
    CONTEXT_SIZE = 4
    BATCH_SIZE = 2
    EMBEDDING_DIM = 16
    STRIDE = 4

    texto = (
        "O cão correu no parque. O gato dormiu no sofá. "
        "Cães e gatos são animais de estimação comuns. "
        "Muitas famílias possuem cães ou gatos em casa."
    )

    print("### 1. Texto original ###")
    print(texto)
    print()

    # ---- Tokenização + Vocabulário + Token IDs ----
    vocab = Vocabulary.from_text(texto)
    tokenizer = SimpleTokenizer(vocab)
    token_ids = tokenizer.encode(texto)

    print(f"### 2. Tokenização e Vocabulário ###")
    print(f"Tamanho do vocabulário: {len(vocab)}")
    print(f"Total de Token IDs gerados: {len(token_ids)}")
    print(f"Primeiros 10 Token IDs: {token_ids[:10]}")
    print()

    # ---- Sequências de treinamento (Dataset/DataLoader) ----
    dataloader = create_dataloader(
        token_ids,
        batch_size=BATCH_SIZE,
        context_size=CONTEXT_SIZE,
        stride=STRIDE,
        shuffle=False,
    )

    print(f"### 3. Sequências de treinamento (DataLoader) ###")
    print(f"Quantidade de lotes: {len(dataloader)}")
    inputs, targets = next(iter(dataloader))
    print(f"Forma do lote de entrada: {tuple(inputs.shape)}")
    print(f"Forma do lote de alvo   : {tuple(targets.shape)}")
    print(f"Exemplo de entrada:\n{inputs}")
    print(f"Exemplo de alvo:\n{targets}")
    print()

    # ---- Embeddings + Positional Embeddings ----
    token_emb_layer = build_token_embedding_layer(len(vocab), EMBEDDING_DIM)
    pos_emb_layer = build_positional_embedding_layer(CONTEXT_SIZE, EMBEDDING_DIM)

    input_embeddings = compute_input_embeddings(inputs, token_emb_layer, pos_emb_layer)

    print(f"### 4. Embeddings + Positional Embeddings ###")
    print(f"Forma dos Token Embeddings (antes da soma): {tuple(token_emb_layer(inputs).shape)}")
    print(f"Forma final (entrada do modelo): {tuple(input_embeddings.shape)}")
    print()
    print("### 5. Entrada final do modelo (pronta para as próximas camadas) ###")
    print(f"Shape: (batch_size={BATCH_SIZE}, context_size={CONTEXT_SIZE}, embedding_dim={EMBEDDING_DIM})")
    print(f"Tensor final:\n{input_embeddings}")


if __name__ == "__main__":
    main()
