"""
embeddings.py

Implementa:
- Token Embedding: converte Token IDs em vetores densos (aprendíveis)
- Positional Embedding: adiciona informação da posição de cada token
  na sequência
- Combinação final: input_embeddings = token_embeddings + positional_embeddings

Este módulo usa PyTorch (nn.Embedding), conforme apresentado no
Capítulo 2. Execute na máquina local, onde o PyTorch já está instalado.
"""

import torch
import torch.nn as nn


def build_token_embedding_layer(vocab_size: int, embedding_dim: int) -> nn.Embedding:
    """
    Cria a camada de embedding de tokens.

    Args:
        vocab_size: quantidade de tokens únicos no vocabulário.
        embedding_dim: dimensão do vetor de representação de cada token.
    """
    torch.manual_seed(123)  # reprodutibilidade
    return nn.Embedding(num_embeddings=vocab_size, embedding_dim=embedding_dim)


def build_positional_embedding_layer(context_size: int, embedding_dim: int) -> nn.Embedding:
    """
    Cria a camada de embedding posicional.

    Args:
        context_size: quantidade máxima de posições (tamanho do contexto).
        embedding_dim: dimensão do vetor de posição (deve ser igual à
            dimensão do token embedding, para que possam ser somados).
    """
    torch.manual_seed(123)
    return nn.Embedding(num_embeddings=context_size, embedding_dim=embedding_dim)


def compute_input_embeddings(
    input_ids: torch.Tensor,
    token_embedding_layer: nn.Embedding,
    positional_embedding_layer: nn.Embedding,
) -> torch.Tensor:
    """
    Combina o embedding de token com o embedding posicional para gerar
    a representação final de entrada do modelo.

    Args:
        input_ids: tensor de forma (batch_size, context_size) com os
            Token IDs de entrada.
        token_embedding_layer: camada de embedding de tokens.
        positional_embedding_layer: camada de embedding posicional.

    Returns:
        Tensor de forma (batch_size, context_size, embedding_dim).
    """
    context_size = input_ids.shape[1]

    token_embeds = token_embedding_layer(input_ids)
    # Gera embeddings para as posições 0, 1, ..., context_size-1
    position_ids = torch.arange(context_size)
    pos_embeds = positional_embedding_layer(position_ids)

    # Broadcasting: (batch, context, dim) + (context, dim)
    input_embeddings = token_embeds + pos_embeds
    return input_embeddings


if __name__ == "__main__":
    # ---- Demonstração ----
    VOCAB_SIZE = 50   # tamanho fictício de vocabulário para o teste
    CONTEXT_SIZE = 4
    EMBEDDING_DIM = 8
    BATCH_SIZE = 2

    token_emb_layer = build_token_embedding_layer(VOCAB_SIZE, EMBEDDING_DIM)
    pos_emb_layer = build_positional_embedding_layer(CONTEXT_SIZE, EMBEDDING_DIM)

    # Lote fictício de Token IDs: (batch_size, context_size)
    input_ids = torch.tensor([
        [2, 6, 5, 13],
        [1, 12, 9, 0],
    ])

    token_embeds = token_emb_layer(input_ids)
    print(f"Forma dos Token Embeddings   : {token_embeds.shape}")

    pos_ids = torch.arange(CONTEXT_SIZE)
    pos_embeds = pos_emb_layer(pos_ids)
    print(f"Forma dos Positional Embeddings: {pos_embeds.shape}")

    input_embeddings = compute_input_embeddings(input_ids, token_emb_layer, pos_emb_layer)
    print(f"Forma da entrada final do modelo: {input_embeddings.shape}")
    print()
    print("Exemplo — vetor final do primeiro token da primeira amostra:")
    print(input_embeddings[0, 0])
