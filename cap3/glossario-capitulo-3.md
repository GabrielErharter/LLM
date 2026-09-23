# Glossário Técnico — Capítulo 3: Attention Mechanisms

> Continuação do glossário cumulativo do projeto (Capítulos 1 e 2).
> Cada termo apresenta: termo original, tradução/equivalente técnico,
> definição, função no modelo, relação com outros conceitos e exemplo
> conceitual/computacional.

---

### Attention (Atenção)

- **Termo original:** Attention
- **Tradução/equivalente:** Mecanismo de atenção
- **Definição:** Mecanismo que permite a um modelo ponderar a importância relativa de diferentes tokens de uma sequência ao construir a representação de cada token, em vez de tratar todos os tokens com o mesmo peso.
- **Função no modelo:** Resolve o problema de dependências de longo alcance — permite que um token "olhe" diretamente para qualquer outro token da sequência, independente da distância entre eles, ao contrário de arquiteturas recorrentes (RNNs).
- **Relação com outros conceitos:** É a base conceitual para Self-Attention, Scaled Dot-Product Attention e Multi-Head Attention; opera sobre os embeddings gerados no Capítulo 2.
- **Exemplo conceitual:** Na frase "O animal não atravessou a rua porque ele estava cansado", a atenção permite ao modelo associar "ele" a "animal", e não a "rua".

---

### Self-Attention

- **Termo original:** Self-Attention
- **Tradução/equivalente:** Autoatenção
- **Definição:** Forma de atenção em que consultas (queries), chaves (keys) e valores (values) são todos derivados da mesma sequência de entrada — o modelo relaciona os tokens de uma sequência entre si.
- **Função no modelo:** Permite que cada token da sequência construa uma representação contextual, considerando a relação com todos os outros tokens da mesma sequência.
- **Relação com outros conceitos:** É a base do Scaled Dot-Product Attention; quando combinada com múltiplas "cabeças" de atenção, dá origem ao Multi-Head Attention.
- **Exemplo conceitual:** Para cada token de "o gato dormiu", o modelo calcula o quanto esse token deve "prestar atenção" em cada um dos outros tokens da mesma frase.

---

### Query, Key, Value (Consulta, Chave, Valor)

- **Termo original:** Query (Q), Key (K), Value (V)
- **Tradução/equivalente:** Consulta, Chave, Valor
- **Definição:** Três projeções lineares distintas do mesmo embedding de entrada, obtidas multiplicando o vetor de entrada por três matrizes de pesos aprendíveis (Wq, Wk, Wv).
- **Função no modelo:** A Query representa "o que este token está buscando"; a Key representa "o que este token oferece" para ser comparado; o Value representa "a informação que será efetivamente repassada" quando a atenção for aplicada.
- **Relação com outros conceitos:** São os componentes fundamentais do cálculo de Scaled Dot-Product Attention — o produto escalar entre Query e Key determina os pesos de atenção, que são aplicados sobre os Values.
- **Exemplo conceitual:** É como uma busca em um dicionário: a Query é o termo buscado, as Keys são as entradas do dicionário comparadas com a busca, e os Values são os conteúdos retornados, ponderados pela relevância da comparação.

---

### Scaled Dot-Product Attention

- **Termo original:** Scaled Dot-Product Attention
- **Tradução/equivalente:** Atenção por produto escalar escalonado
- **Definição:** Mecanismo de atenção que calcula os pesos através do produto escalar entre Query e Key, divide o resultado pela raiz quadrada da dimensão das chaves (escalonamento) e aplica a função Softmax para obter uma distribuição de probabilidade sobre os tokens.
- **Função no modelo:** O escalonamento evita que os produtos escalares cresçam demais em dimensões altas, o que levaria a gradientes muito pequenos após o Softmax (saturação), prejudicando o treinamento.
- **Relação com outros conceitos:** É a formalização matemática da Self-Attention; a fórmula é `Attention(Q,K,V) = softmax(QKᵀ / √d_k) V`, onde `d_k` é a dimensão das chaves.
- **Exemplo conceitual:** Sem o fator de escala, em dimensões grandes (ex: d_k=512), os valores de QKᵀ podem ficar muito altos, fazendo o Softmax se aproximar de uma função "tudo ou nada" — o escalonamento suaviza essa distribuição.

---

### Softmax

- **Termo original:** Softmax
- **Tradução/equivalente:** Função Softmax (sem tradução direta comum)
- **Definição:** Função matemática que converte um vetor de números reais em uma distribuição de probabilidade, onde todos os valores somam 1 e são não-negativos.
- **Função no modelo:** Transforma os "scores" brutos de atenção (produtos escalares) em pesos de atenção interpretáveis como percentuais de importância entre os tokens.
- **Relação com outros conceitos:** É aplicada sobre os scores calculados em Scaled Dot-Product Attention, logo após a etapa de escalonamento (e, na Causal Attention, após a máscara).
- **Exemplo conceitual:** Scores `[2.0, 1.0, 0.1]` após Softmax podem virar aproximadamente `[0.66, 0.24, 0.10]` — valores entre 0 e 1 que somam 1.

---

### Causal Attention (Atenção Causal / Mascarada)

- **Termo original:** Causal Attention / Masked Attention
- **Tradução/equivalente:** Atenção causal (ou mascarada)
- **Definição:** Variante da atenção em que cada token só pode "ver" e utilizar informações dos tokens anteriores a ele (e ele mesmo), impedindo o acesso a tokens futuros na sequência.
- **Função no modelo:** É essencial para modelos autorregressivos como o GPT, que preveem o próximo token com base apenas no contexto anterior — sem essa restrição, o modelo "trapacearia" durante o treinamento, olhando para a resposta que deveria prever.
- **Relação com outros conceitos:** É implementada aplicando uma máscara (geralmente com valores `-infinito`) na matriz de scores antes do Softmax, zerando efetivamente a atenção para posições futuras.
- **Exemplo conceitual:** Ao processar o 3º token de uma sequência de 5, a máscara causal impede que ele "veja" os tokens nas posições 4 e 5, mesmo que eles estejam presentes na sequência de entrada.

---

### Dropout (aplicado à Attention)

- **Termo original:** Dropout
- **Tradução/equivalente:** Descarte aleatório (regularização)
- **Definição:** Técnica de regularização que zera aleatoriamente uma fração dos pesos de atenção durante o treinamento, evitando que o modelo dependa excessivamente de conexões específicas.
- **Função no modelo:** Reduz o overfitting, forçando o modelo a não depender de padrões de atenção muito específicos, tornando a representação aprendida mais robusta.
- **Relação com outros conceitos:** É aplicado tipicamente após o Softmax, sobre os pesos de atenção, antes da multiplicação pelos Values.
- **Exemplo conceitual:** Com dropout de 10%, aproximadamente 1 em cada 10 pesos de atenção é zerado aleatoriamente a cada passo de treinamento.

---

### Multi-Head Attention

- **Termo original:** Multi-Head Attention
- **Tradução/equivalente:** Atenção com múltiplas cabeças
- **Definição:** Extensão do mecanismo de atenção que executa várias operações de Scaled Dot-Product Attention em paralelo (as "heads"), cada uma com suas próprias matrizes de projeção (Wq, Wk, Wv), e depois concatena e projeta os resultados.
- **Função no modelo:** Permite que o modelo aprenda diferentes tipos de relações entre os tokens simultaneamente — uma head pode aprender relações sintáticas, outra relações semânticas, por exemplo.
- **Relação com outros conceitos:** Cada "head" é uma instância de Causal/Scaled Dot-Product Attention operando sobre uma fatia menor da dimensão de embedding (`d_model / num_heads`); os resultados de todas as heads são concatenados e passam por uma projeção linear final.
- **Exemplo conceitual:** Com `d_model=512` e `num_heads=8`, cada head opera com dimensão `64` (512/8); os 8 resultados de dimensão 64 são concatenados de volta para 512 e projetados novamente.

---

### Matriz de Atenção (Attention Matrix / Attention Weights)

- **Termo original:** Attention Weights / Attention Matrix
- **Tradução/equivalente:** Matriz (ou pesos) de atenção
- **Definição:** Matriz quadrada de dimensão (context_size × context_size) que armazena, para cada par de tokens, o peso de atenção que um token atribui ao outro, após o Softmax.
- **Função no modelo:** É o resultado intermediário que efetivamente determina como a informação dos Values é combinada para formar a saída de cada posição.
- **Relação com outros conceitos:** É o principal artefato usado para visualização e interpretação do comportamento do mecanismo de atenção; na Causal Attention, sua metade superior é zerada pela máscara.
- **Exemplo conceitual:** Numa sequência de 4 tokens, a matriz de atenção tem forma (4,4); a célula `[2][0]` indica o quanto o token na posição 2 "presta atenção" no token da posição 0.

---

*Glossário complementar aos Capítulos 1 e 2. Consolida os conceitos centrais do mecanismo de Attention utilizado na arquitetura Transformer.*
