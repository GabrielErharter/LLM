"""
tokenizer.py

Implementa:
- Tokenização de texto em tokens (palavras e pontuação)
- Construção de vocabulário (token -> id e id -> token)
- Codificação (texto -> Token IDs) e decodificação (Token IDs -> texto)
- Tratamento de tokens especiais: <|unk|> e <|endoftext|>

Baseado no fluxo apresentado no Capítulo 2 (Working with Text Data).
"""

import re


def basic_tokenize(text: str) -> list[str]:
    """
    Divide um texto bruto em uma lista de tokens (palavras e sinais de
    pontuação), separando espaços em branco.

    Exemplo:
        "Olá, mundo!" -> ["Olá", ",", "mundo", "!"]
    """
    # Separa em pontuação (,.:;?_!"()') e espaços, mantendo os símbolos
    pieces = re.split(r'([,.:;?_!"()\']|--|\s)', text)
    # Remove strings vazias e espaços em branco "puros"
    tokens = [item.strip() for item in pieces if item.strip()]
    return tokens


class Vocabulary:
    """
    Representa o vocabulário do modelo: o conjunto de tokens únicos
    conhecidos, cada um associado a um Token ID.
    """

    UNK_TOKEN = "<|unk|>"
    EOT_TOKEN = "<|endoftext|>"

    def __init__(self, tokens: list[str]):
        # Ordena para ter um vocabulário determinístico e reprodutível
        unique_tokens = sorted(set(tokens))
        # Adiciona os tokens especiais ao final do vocabulário
        unique_tokens.extend([self.UNK_TOKEN, self.EOT_TOKEN])

        self.token_to_id = {token: idx for idx, token in enumerate(unique_tokens)}
        self.id_to_token = {idx: token for token, idx in self.token_to_id.items()}

    def __len__(self):
        return len(self.token_to_id)

    @classmethod
    def from_text(cls, text: str) -> "Vocabulary":
        """Constrói um vocabulário a partir de um texto de treinamento."""
        tokens = basic_tokenize(text)
        return cls(tokens)


class SimpleTokenizer:
    """
    Tokenizador que combina a tokenização básica com um vocabulário,
    permitindo converter texto <-> Token IDs.
    """

    def __init__(self, vocab: Vocabulary):
        self.vocab = vocab

    def encode(self, text: str) -> list[int]:
        """Converte um texto em uma lista de Token IDs."""
        tokens = basic_tokenize(text)
        # Substitui tokens desconhecidos pelo token especial <|unk|>
        tokens = [
            token if token in self.vocab.token_to_id else Vocabulary.UNK_TOKEN
            for token in tokens
        ]
        return [self.vocab.token_to_id[token] for token in tokens]

    def decode(self, ids: list[int]) -> str:
        """Converte uma lista de Token IDs de volta em texto."""
        tokens = [self.vocab.id_to_token[i] for i in ids]
        text = " ".join(tokens)
        # Remove espaços indevidos antes de pontuação
        text = re.sub(r'\s+([,.:;?_!"()\'])', r'\1', text)
        return text


if __name__ == "__main__":
    # ---- Teste rápido com diferentes frases ----
    corpus = (
        "O cão correu no parque. O gato dormiu no sofá. "
        "Cães e gatos são animais de estimação comuns."
    )

    vocab = Vocabulary.from_text(corpus)
    tokenizer = SimpleTokenizer(vocab)

    frases_teste = [
        "O cão correu no parque.",
        "O papagaio voou alto!",  # contém palavra fora do vocabulário
        "Cães, gatos e papagaios.",
    ]

    print(f"Tamanho do vocabulário: {len(vocab)}\n")

    for frase in frases_teste:
        tokens = basic_tokenize(frase)
        ids = tokenizer.encode(frase)
        decoded = tokenizer.decode(ids)
        print(f"Frase original : {frase}")
        print(f"Tokens         : {tokens}")
        print(f"Token IDs      : {ids}")
        print(f"Decodificado   : {decoded}")
        print("-" * 60)
