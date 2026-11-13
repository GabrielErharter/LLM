"""
tokenizador.py (sem dependencia de PyTorch)

Dois tokenizadores simples, com a mesma interface:
- TokenizadorCaractere: 1 token = 1 caractere (vocabulario pequeno, treina
  rapido em CPU e nao tem tokens desconhecidos).
- TokenizadorPalavra: 1 token = palavra ou pontuacao (igual a Sprint 1),
  com o token especial <|unk|> para palavras fora do vocabulario.

Interface: encode(texto) -> ids, decode(ids) -> texto, len(tok),
.vocab (lista), .tipo, .info() (dicionario salvavel no checkpoint).
"""
import re

UNK = "<|unk|>"
_PADRAO = re.compile(r'([,.:;?_!"()\']|--|\s)')


class TokenizadorCaractere:
    tipo = "char"

    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.t2i = {c: i for i, c in enumerate(self.vocab)}
        self.i2t = {i: c for c, i in self.t2i.items()}

    @classmethod
    def de_texto(cls, texto):
        return cls(sorted(set(texto)))

    def __len__(self):
        return len(self.vocab)

    def encode(self, texto):
        # caracteres fora do vocabulario sao ignorados
        return [self.t2i[c] for c in texto if c in self.t2i]

    def decode(self, ids):
        return "".join(self.i2t[int(i)] for i in ids)

    def info(self):
        return {"tipo": self.tipo, "vocab": self.vocab}


class TokenizadorPalavra:
    tipo = "palavra"

    def __init__(self, vocab):
        self.vocab = list(vocab)
        self.t2i = {t: i for i, t in enumerate(self.vocab)}
        self.i2t = {i: t for t, i in self.t2i.items()}

    @staticmethod
    def _tokenizar(texto):
        return [p for p in _PADRAO.split(texto) if p and not p.isspace()]

    @classmethod
    def de_texto(cls, texto):
        return cls(sorted(set(cls._tokenizar(texto))) + [UNK])

    def __len__(self):
        return len(self.vocab)

    def encode(self, texto):
        unk = self.t2i[UNK]
        return [self.t2i.get(t, unk) for t in self._tokenizar(texto)]

    def decode(self, ids):
        texto = " ".join(self.i2t[int(i)] for i in ids)
        return re.sub(r'\s+([,.:;?_!"()\'])', r'\1', texto)

    def info(self):
        return {"tipo": self.tipo, "vocab": self.vocab}


_TIPOS = {"char": TokenizadorCaractere, "palavra": TokenizadorPalavra}


def criar_tokenizador(tipo, texto):
    """Constroi o vocabulario a partir do texto."""
    return _TIPOS[tipo].de_texto(texto)


def tokenizador_de_info(info):
    """Reconstroi um tokenizador a partir do dicionario salvo no checkpoint."""
    return _TIPOS[info["tipo"]](info["vocab"])
