# Glossário Técnico — Capítulo 5: Pretraining on Unlabeled Data

> Continuação do glossário cumulativo (Capítulos 1 a 4).
> Notação: B = lote, T = tokens por sequência, V = tamanho do vocabulário.

---

### Função de perda (Loss Function)

- **Termo original:** Loss function
- **Tradução/equivalente:** Função de perda / função de custo
- **Definição:** Função que associa um número à diferença entre as previsões do modelo e os valores esperados. Quanto menor, melhor o modelo se ajusta à tarefa.
- **Função no treinamento:** É o valor que o treinamento tenta minimizar; seu gradiente indica em que direção ajustar os pesos.
- **Relação com outros conceitos:** No GPT, a perda usada é a Cross-Entropy sobre os logits; a Perplexity é derivada dela; o otimizador usa seus gradientes para atualizar os pesos.
- **Exemplo conceitual:** Uma loss de 4,0 no início e de 1,5 depois de treinar indica que o modelo passou a atribuir probabilidades maiores aos tokens corretos.

---

### Cross-Entropy (Entropia Cruzada)

- **Termo original:** Cross-entropy loss
- **Tradução/equivalente:** Entropia cruzada
- **Definição:** Média, sobre todas as posições, do negativo do logaritmo da probabilidade que o modelo atribuiu ao token correto: `L = −(1/N) Σ log p(token_correto)`. Internamente aplica Softmax aos logits.
- **Função no treinamento:** Mede o quão "surpreso" o modelo fica com o próximo token real. Penaliza fortemente previsões confiantes e erradas.
- **Relação com outros conceitos:** Recebe logits `(B,T,V)` achatados para `(B·T, V)` e alvos `(B·T)` (os alvos são a entrada deslocada em 1 posição, Sprint 1). Seu gradiente em relação aos logits é `softmax − one-hot`.
- **Exemplo conceitual:** Um modelo que chuta uniformemente entre V tokens tem `L = ln(V)`; com V = 52, `L ≈ 3,95`. Esse é o ponto de partida esperado de um modelo não treinado.

---

### Perplexity (Perplexidade)

- **Termo original:** Perplexity
- **Tradução/equivalente:** Perplexidade
- **Definição:** Exponencial da cross-entropy: `PPL = exp(L)`. Pode ser lida como o número efetivo de tokens igualmente prováveis entre os quais o modelo "hesita" a cada passo.
- **Função no treinamento:** Torna a loss mais interpretável e comparável: perplexity menor significa que o modelo prevê melhor o próximo token.
- **Relação com outros conceitos:** Deriva diretamente da Cross-Entropy; um modelo aleatório tem `PPL = V`, e um modelo perfeito tem `PPL = 1`.
- **Exemplo conceitual:** Loss 2,0 → perplexity ≈ 7,4: é como se o modelo estivesse escolhendo entre ~7 tokens equiprováveis a cada posição.

---

### Pré-treinamento (Pretraining) e Treinamento

- **Termo original:** Pretraining / training
- **Tradução/equivalente:** Pré-treinamento / treinamento
- **Definição:** Processo de ajustar os pesos do modelo para prever o próximo token em um grande texto sem rótulos (aprendizado autossupervisionado: o alvo é o próprio texto deslocado).
- **Função no treinamento:** Faz o modelo absorver regularidades da linguagem (ortografia, sintaxe, padrões de estilo), tornando-o útil para geração e para um posterior fine-tuning.
- **Relação com outros conceitos:** Usa o Dataset/DataLoader (Sprint 1), o GPT (Sprint 4), a Cross-Entropy e um otimizador.
- **Exemplo conceitual:** Dado `"O rio cor"`, o alvo em cada posição é o caractere seguinte: `"O rio corr"`.

---

### Época, Lote e Passo (Epoch, Batch, Step)

- **Termo original:** Epoch, batch, step (iteration)
- **Tradução/equivalente:** Época, lote, passo (iteração)
- **Definição:** *Lote* é o grupo de sequências processado de uma vez; *passo* é uma atualização de pesos (um lote); *época* é uma passagem completa por todo o conjunto de treino.
- **Função no treinamento:** Organizam o laço de treino: `passos por época = nº de lotes`, e `passos totais = épocas × lotes`.
- **Relação com outros conceitos:** Lote maior → menos passos por época e gradiente menos ruidoso; mais épocas → mais atualizações (e maior risco de Sobreajuste).
- **Exemplo conceitual:** 2.000 janelas de treino com lote 16 geram 125 passos por época.

---

### Gradiente e Retropropagação (Backpropagation)

- **Termo original:** Gradient, backpropagation
- **Tradução/equivalente:** Gradiente, retropropagação
- **Definição:** O gradiente é o vetor das derivadas da loss em relação a cada peso; a retropropagação calcula esses gradientes de trás para frente na rede (regra da cadeia). Em PyTorch: `loss.backward()`.
- **Função no treinamento:** Indica, para cada peso, se aumentá-lo ou diminuí-lo reduz a loss.
- **Relação com outros conceitos:** Alimenta o otimizador; as conexões residuais e a LayerNorm (Cap. 4) ajudam o gradiente a atravessar muitas camadas. Deve ser zerado a cada passo (`optimizer.zero_grad()`), pois o PyTorch acumula.
- **Exemplo conceitual:** Para a cross-entropy, o gradiente nos logits é `softmax − one-hot`: empurra para cima o logit do token correto e para baixo os demais.

---

### Otimizador (Optimizer) e AdamW

- **Termo original:** Optimizer, AdamW
- **Tradução/equivalente:** Otimizador
- **Definição:** Algoritmo que usa os gradientes para atualizar os pesos. O *AdamW* adapta o passo de cada parâmetro com médias móveis dos gradientes e aplica *weight decay* de forma desacoplada.
- **Função no treinamento:** Executa a atualização (`optimizer.step()`); define quão rápido e estável é a convergência.
- **Relação com outros conceitos:** Depende da Taxa de Aprendizado e do Weight Decay; guarda estado interno (médias móveis), que por isso é salvo no checkpoint para retomar o treino.
- **Exemplo conceitual:** `torch.optim.AdamW(model.parameters(), lr=5e-4, weight_decay=0.1)`.

---

### Taxa de aprendizado (Learning Rate)

- **Termo original:** Learning rate
- **Tradução/equivalente:** Taxa de aprendizado
- **Definição:** Hiperparâmetro que escala o tamanho de cada atualização de pesos.
- **Função no treinamento:** Controla o equilíbrio entre velocidade e estabilidade: muito baixa → convergência lenta; muito alta → oscilação ou divergência da loss.
- **Relação com outros conceitos:** Atua junto com o otimizador e o tamanho do lote; é um dos principais parâmetros a investigar nos experimentos.
- **Exemplo conceitual:** Comparar `1e-4`, `5e-4`, `2e-3` e `1e-2` e observar qual curva de loss desce mais rápido sem instabilidade.

---

### Weight Decay (Decaimento de pesos)

- **Termo original:** Weight decay
- **Tradução/equivalente:** Decaimento de pesos (regularização L2 desacoplada)
- **Definição:** Penalização que puxa os pesos levemente para zero a cada passo.
- **Função no treinamento:** Regulariza o modelo, desencorajando pesos grandes e ajudando a reduzir o Sobreajuste.
- **Relação com outros conceitos:** Parâmetro do AdamW; complementa o Dropout.
- **Exemplo conceitual:** `weight_decay=0.1`.

---

### Conjuntos de Treino e Validação

- **Termo original:** Training set, validation set
- **Tradução/equivalente:** Conjunto de treino, conjunto de validação
- **Definição:** O texto é dividido em duas partes: o modelo aprende no *treino* e é avaliado no *validação*, que ele nunca usa para atualizar pesos.
- **Função no treinamento:** A loss de validação estima o desempenho em dados novos; comparar as duas losses revela o Sobreajuste.
- **Relação com outros conceitos:** A divisão é feita por posição (ex.: 90% / 10%), sem embaralhar antes, para que nenhuma janela de validação compartilhe tokens com o treino.
- **Exemplo conceitual:** Treino 2,0 → 1,0 enquanto validação 2,0 → 1,6 indica que o modelo generaliza pior do que memoriza.

---

### Sobreajuste (Overfitting)

- **Termo original:** Overfitting
- **Tradução/equivalente:** Sobreajuste
- **Definição:** Situação em que o modelo se ajusta tão bem aos dados de treino (inclusive ao ruído e a trechos específicos) que perde capacidade de generalizar.
- **Função no treinamento:** É o principal sinal a monitorar para decidir quando parar de treinar ou quando regularizar mais.
- **Relação com outros conceitos:** Sinais: loss de treino cai enquanto a de validação estabiliza ou sobe. Mitigação: Dropout, Weight Decay, mais dados, menos épocas. Textos pequenos favorecem o fenômeno.
- **Exemplo conceitual:** Loss de validação mínima na época 4 e crescendo da época 5 em diante, com a de treino ainda caindo.

---

### Convergência

- **Termo original:** Convergence
- **Tradução/equivalente:** Convergência
- **Definição:** Estado em que a loss deixa de melhorar de forma relevante conforme o treino continua.
- **Função no treinamento:** Indica que o modelo atingiu o que consegue aprender com aquela configuração e aqueles dados.
- **Relação com outros conceitos:** Depende da taxa de aprendizado, do tamanho do modelo, da quantidade de dados e do número de épocas.
- **Exemplo conceitual:** Curva de loss de validação que se achata a partir de certa época.

---

### Gradient Clipping (Recorte de gradiente)

- **Termo original:** Gradient clipping
- **Tradução/equivalente:** Recorte de gradiente
- **Definição:** Limita a norma do vetor de gradientes a um valor máximo antes da atualização dos pesos.
- **Função no treinamento:** Evita passos muito grandes (instabilidade) quando o gradiente "explode".
- **Relação com outros conceitos:** Aplicado entre `loss.backward()` e `optimizer.step()`; é opcional e fica desligado na configuração base.
- **Exemplo conceitual:** `torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)`.

---

### Checkpoint e `state_dict`

- **Termo original:** Checkpoint, state_dict
- **Tradução/equivalente:** Ponto de salvamento / dicionário de estado
- **Definição:** O `state_dict` é o dicionário que mapeia o nome de cada camada aos seus tensores de pesos; o checkpoint é o arquivo que o guarda, junto com o estado do otimizador, a configuração da arquitetura e o vocabulário.
- **Função no treinamento:** Permite preservar o resultado do treino, retomá-lo depois e reproduzir a inferência sem retreinar.
- **Relação com outros conceitos:** Salvar com `torch.save` e restaurar com `model.load_state_dict`; é preciso reconstruir a mesma arquitetura (por isso a configuração vai no checkpoint) e o mesmo tokenizador.
- **Exemplo conceitual:** Depois de carregar, o modelo deve produzir exatamente os mesmos logits que o modelo em memória.

---

### Decodificação Gulosa (Greedy Decoding)

- **Termo original:** Greedy decoding
- **Tradução/equivalente:** Decodificação gulosa
- **Definição:** A cada passo, escolhe o token de maior probabilidade (`argmax`).
- **Função na geração:** Produz saída determinística e estável, mas tende a repetir trechos e a ser pouco variada.
- **Relação com outros conceitos:** É o caso limite da amostragem com temperatura → 0 e equivale a `top_k = 1`.
- **Exemplo conceitual:** Mesmo prompt, mesma saída em toda execução.

---

### Amostragem Probabilística, Temperatura e Top-k

- **Termo original:** Multinomial sampling, temperature, top-k sampling
- **Tradução/equivalente:** Amostragem multinomial, temperatura, amostragem top-k
- **Definição:** Em vez do `argmax`, sorteia-se o próximo token de acordo com as probabilidades (softmax). A *temperatura* divide os logits antes do softmax (T < 1 concentra, T > 1 achata a distribuição); o *top-k* zera a probabilidade de todos os tokens fora dos k mais prováveis.
- **Função na geração:** Controlam o equilíbrio entre coerência (T baixa, k pequeno) e diversidade (T alta, k grande); evitam o loop típico do greedy.
- **Relação com outros conceitos:** Atuam sobre os logits produzidos pelo modelo; seu efeito é independente do treino, mas só gera texto bom se os pesos forem bons. Em um modelo não treinado, a distribuição é quase uniforme e nenhuma estratégia produz texto coerente.
- **Exemplo conceitual:** `T=0,5, top_k=5`: sorteia entre os 5 tokens mais prováveis, com a distribuição ainda mais concentrada nos primeiros.

---

### Baseline

- **Termo original:** Baseline
- **Tradução/equivalente:** Linha de base / referência
- **Definição:** Modelo simples usado como régua de comparação. Aqui: o chute uniforme (`loss = ln V`) e um modelo estatístico de bigramas (`baseline_bigrama.py`).
- **Função na avaliação:** Permite saber se o GPT aprende de fato mais do que estatística trivial.
- **Relação com outros conceitos:** Comparado com a loss e a perplexity de validação do GPT.
- **Exemplo conceitual:** Se o GPT tem loss de validação acima da do bigrama, ainda não aprendeu o suficiente (ou está sobreajustado, ou os dados são pequenos demais).

---

*Glossário complementar aos Capítulos 1 a 4.*
