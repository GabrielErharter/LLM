"""
sequences.py

Implementa a preparação das sequências de treinamento a partir de uma
lista de Token IDs, usando o mecanismo de janela deslizante (sliding
window) apresentado no Capítulo 2.

Para cada posição da janela, gera:
- input_ids : sequência de tokens de entrada
- target_ids: mesma sequência deslocada em 1 posição (o token seguinte
              a cada posição de entrada)
"""


def create_input_target_pairs(
    token_ids: list[int], context_size: int, stride: int = 1
) -> list[tuple[list[int], list[int]]]:
    """
    Gera pares (entrada, alvo) usando uma janela deslizante.

    Args:
        token_ids: lista completa de Token IDs do texto.
        context_size: tamanho da janela de contexto (quantos tokens
            o modelo "vê" de cada vez).
        stride: passo do deslizamento da janela a cada nova amostra.

    Returns:
        Lista de tuplas (input_ids, target_ids), cada uma com
        `context_size` tokens.
    """
    pairs = []
    for i in range(0, len(token_ids) - context_size, stride):
        input_ids = token_ids[i: i + context_size]
        target_ids = token_ids[i + 1: i + context_size + 1]
        pairs.append((input_ids, target_ids))
    return pairs


if __name__ == "__main__":
    # Sequência de exemplo simulando Token IDs já gerados na etapa anterior
    exemplo_ids = list(range(1, 21))  # [1, 2, ..., 20]

    print("Sequência de entrada:", exemplo_ids)
    print()

    for context_size in [4, 8]:
        pares = create_input_target_pairs(exemplo_ids, context_size, stride=context_size)
        print(f"--- context_size={context_size}, stride={context_size} ---")
        for entrada, alvo in pares:
            print(f"entrada={entrada}  ->  alvo={alvo}")
        print(f"Total de amostras geradas: {len(pares)}\n")
