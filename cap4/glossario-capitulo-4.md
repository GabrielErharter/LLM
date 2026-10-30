# Glossário Técnico — Capítulo 4: Implementing a GPT Model

> Continuação do glossário cumulativo (Capítulos 1, 2 e 3).
> Notação: B = batch, T = tokens na sequência, D = dimensão do embedding, V = tamanho do vocabulário.

---

### Transformer

- **Termo original:** Transformer
- **Tradução/equivalente:** Arquitetura Transformer
- **Definição:** Arquitetura de rede neural baseada em mecanismos de atenção, que processa todos os tokens de uma sequência em paralelo e os relaciona entre si, sem recorrência.
- **Função no modelo:** É a base estrutural do GPT: o modelo é uma pilha de blocos Transformer sobre os embeddings de entrada.
- **Relação com outros conceitos:** É composta de Transformer Blocks, que combinam Multi-Head Attention (Cap. 3), LayerNorm, FeedForward e conexões residuais. O GPT usa apenas a parte "decoder" da arquitetura original.
- **Exemplo conceitual:** Uma sequência `(B, T, D)` entra e sai do Transformer com a mesma forma; o que muda é que cada vetor passa a incorporar o contexto dos tokens anteriores.

---

### Transformer Block (Bloco Transformer)

- **Termo original:** Transformer Block
- **Tradução/equivalente:** Bloco Transformer
- **Definição:** Unidade repetível da arquitetura, formada por dois sub-blocos: (1) LayerNorm → Multi-Head Attention → Dropout → soma residual; (2) LayerNorm → FeedForward → Dropout → soma residual.
- **Função no modelo:** Refina a representação de cada token: a atenção mistura informação entre tokens; a FeedForward transforma cada token individualmente.
- **Relação com outros conceitos:** Entrada e saída têm a mesma forma `(B, T, D)`, o que permite empilhar N blocos. Integra todos os componentes deste capítulo.
- **Exemplo conceitual:** O GPT-2 pequeno empilha 12 blocos idênticos em estrutura (com pesos diferentes).

---

### Layer Normalization (LayerNorm)

- **Termo original:** Layer Normalization
- **Tradução/equivalente:** Normalização de camada
- **Definição:** Normaliza o vetor de cada token (última dimensão) para média 0 e variância 1, e depois aplica uma escala (`scale`) e um deslocamento (`shift`) aprendíveis. Um pequeno `eps` evita divisão por zero.
- **Função no modelo:** Estabiliza a escala das ativações entre camadas, facilitando o treinamento de redes profundas.
- **Relação com outros conceitos:** Aplicada antes da atenção e antes da FeedForward em cada bloco (arquitetura *pre-LayerNorm*), e uma vez mais após o último bloco (normalização final).
- **Exemplo conceitual:** Um vetor com média 7 e desvio 3 sai com média ≈ 0 e desvio ≈ 1 (antes de `scale`/`shift`).

---

### Conexão Residual (Residual / Shortcut Connection)

- **Termo original:** Residual (shortcut) connection
- **Tradução/equivalente:** Conexão residual / atalho
- **Definição:** Soma da entrada de um sub-bloco à sua saída: `y = x + f(x)`.
- **Função no modelo:** Cria um caminho direto para o gradiente atravessar as camadas, atenuando o problema do desaparecimento de gradiente em redes profundas; também permite que o sub-bloco aprenda apenas uma "correção" à entrada.
- **Relação com outros conceitos:** Presente duas vezes por Transformer Block (após a atenção e após a FeedForward). Exige que entrada e saída do sub-bloco tenham a mesma forma.
- **Exemplo conceitual:** Se a atenção devolver zeros, o bloco simplesmente repassa `x` adiante.

---

### FeedForward Network (Rede Feed-Forward)

- **Termo original:** FeedForward Network (FFN / MLP)
- **Tradução/equivalente:** Rede alimentada adiante
- **Definição:** Duas camadas lineares com uma ativação não linear no meio: `Linear(D → 4D) → GELU → Linear(4D → D)`, aplicada a cada token de forma independente.
- **Função no modelo:** Adiciona capacidade não linear e transforma a representação de cada token após a atenção; é onde fica a maior parte dos parâmetros do bloco.
- **Relação com outros conceitos:** Usa GELU como ativação; recebe a saída normalizada pela segunda LayerNorm; seu resultado entra na segunda conexão residual.
- **Exemplo conceitual:** Com D = 128, a camada interna tem 512 neurônios; a FFN responde por cerca de 67% dos parâmetros de cada bloco.

---

### GELU (Gaussian Error Linear Unit)

- **Termo original:** GELU
- **Tradução/equivalente:** Unidade linear de erro gaussiano
- **Definição:** Função de ativação suave, definida como `x · Φ(x)`, onde Φ é a CDF da normal padrão; usa-se a aproximação com `tanh`.
- **Função no modelo:** Introduz não linearidade na FeedForward. Diferente da ReLU, é suave e deixa passar pequenos valores negativos, o que costuma melhorar o fluxo de gradiente.
- **Relação com outros conceitos:** É a ativação da FeedForward Network.
- **Exemplo conceitual:** `GELU(0) = 0`, `GELU(3) ≈ 2,996`, `GELU(-3) ≈ -0,004`.

---

### Dropout

- **Termo original:** Dropout
- **Tradução/equivalente:** Descarte aleatório (regularização)
- **Definição:** Durante o treinamento, zera aleatoriamente uma fração das ativações (taxa `drop_rate`), reescalando as demais.
- **Função no modelo:** Reduz overfitting. É desativado na inferência (`model.eval()`), o que torna a saída determinística.
- **Relação com outros conceitos:** Aplicado nos embeddings, nos pesos de atenção e nas saídas dos sub-blocos (antes da soma residual).
- **Exemplo conceitual:** Com `drop_rate=0.1`, cerca de 10% das ativações são zeradas a cada passagem de treino.

---

### Logits

- **Termo original:** Logits
- **Tradução/equivalente:** Logits (pontuações brutas)
- **Definição:** Vetor de pontuações reais, sem normalização, produzido pelo modelo para cada posição, com um valor por token do vocabulário. Forma `(B, T, V)`.
- **Função no modelo:** São a saída do GPT; indicam o quanto cada token do vocabulário é plausível como próximo token.
- **Relação com outros conceitos:** Aplicando Softmax obtêm-se probabilidades; na geração, usa-se apenas o logit da última posição. São produzidos pela Language Model Head.
- **Exemplo conceitual:** Com V = 1000, cada posição produz 1000 números; o maior indica o token mais provável.

---

### Language Model Head (Cabeça de Linguagem / Camada de Saída)

- **Termo original:** Language Model Head (output head)
- **Tradução/equivalente:** Cabeça do modelo de linguagem
- **Definição:** Camada linear final `Linear(D → V)` (sem bias, no GPT-2) que projeta cada vetor de dimensão D para um vetor de logits de tamanho V.
- **Função no modelo:** Converte a representação contextual de cada token em pontuações sobre o vocabulário.
- **Relação com outros conceitos:** Vem depois da LayerNorm final; seu custo cresce com `D × V`, podendo dominar o número de parâmetros em vocabulários grandes. No GPT-2 original compartilha pesos com o embedding de tokens (*weight tying*).
- **Exemplo conceitual:** Com D = 768 e V = 50.257, a cabeça tem ≈ 38,6 milhões de parâmetros.

---

### Inferência

- **Termo original:** Inference
- **Tradução/equivalente:** Inferência
- **Definição:** Uso do modelo já construído (ou treinado) para produzir saídas, sem atualizar pesos, em modo `eval()` e normalmente sob `torch.no_grad()`.
- **Função no modelo:** É a fase em que o modelo gera texto a partir de um prompt.
- **Relação com outros conceitos:** Desativa o Dropout; é executada repetidamente na geração autorregressiva.
- **Exemplo conceitual:** `model.eval()` + `torch.no_grad()` + `model(tokens)`.

---

### Geração Autorregressiva

- **Termo original:** Autoregressive generation
- **Tradução/equivalente:** Geração autorregressiva
- **Definição:** Processo iterativo em que o modelo prevê o próximo token, o anexa ao contexto e repete o processo usando a sequência ampliada.
- **Função no modelo:** Permite gerar textos de tamanho arbitrário a partir de um prompt inicial, um token por vez.
- **Relação com outros conceitos:** Depende da Causal Attention (Cap. 3), que impede o modelo de "ver o futuro", e do recorte do contexto à janela `context_length`. Cada passo: logits → (top-k) → (temperatura) → softmax → seleção → concatenação.
- **Exemplo conceitual:** Prompt `[1, 2, 3]` → prevê 25 → contexto `[1, 2, 3, 25]` → prevê 32 → e assim por diante.

---

### Estratégias de Seleção: Greedy, Temperatura e Top-k

- **Termo original:** Greedy decoding, Temperature, Top-k sampling
- **Tradução/equivalente:** Decodificação gulosa, temperatura, amostragem top-k
- **Definição:** *Greedy* escolhe sempre o token de maior logit (`argmax`). *Temperatura* divide os logits antes do Softmax (`logits / T`): T < 1 concentra a distribuição, T > 1 a achata. *Top-k* mantém só os k logits mais altos (os demais viram `-∞`) antes de sortear.
- **Função no modelo:** Controlam o equilíbrio entre previsibilidade (greedy, T baixa, k pequeno) e diversidade (T alta, k grande).
- **Relação com outros conceitos:** Operam sobre os logits e antecedem a seleção do token na geração autorregressiva.
- **Exemplo conceitual:** Com T → 0 a geração tende ao greedy; com `top_k=1` ela é equivalente ao greedy.

---

### Contagem de Parâmetros

- **Termo original:** Parameter count
- **Tradução/equivalente:** Número de parâmetros
- **Definição:** Total de pesos treináveis do modelo. Para este GPT: `V·D + T_ctx·D` (embeddings) + `N · (4D² + D + 2·D·F + F + D + 4D)` (blocos, com F = dimensão da FFN) + `2D` (LayerNorm final) + `D·V` (cabeça).
- **Função no modelo:** Indica a capacidade e o custo de memória do modelo.
- **Relação com outros conceitos:** Cresce linearmente com o número de blocos, quadraticamente com D, e não depende do número de heads (para D fixo).
- **Exemplo conceitual:** O GPT-2 pequeno (V=50.257, ctx=1024, D=768, N=12) tem ≈ 124 milhões de parâmetros com *weight tying* e ≈ 163 milhões sem.

---

*Glossário complementar aos Capítulos 1, 2 e 3.*
