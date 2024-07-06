- [Space occupancy analysis](#space-occupancy-analysis)
  - [Parameter quantity and model size analysis and testing](#parameter-quantity-and-model-size-analysis-and-testing)
    - [Parameter quantity and model size analysis](#parameter-quantity-and-model-size-analysis)
      - [Hyperparameters](#hyperparameters)
      - [llama2 model structure](#llama2-model-structure)
      - [llama\_7B parameter number](#llama_7b-parameter-number)
    - [Parameter number and model size test](#parameter-number-and-model-size-test)
  - [gpu memory usage analysis and test](#gpu-memory-usage-analysis-and-test)
    - [gpu memory usage analysis](#gpu-memory-usage-analysis)
      - [Model usage analysis](#model-usage-analysis)
      - [Intermediate value (activation value) usage analysis](#intermediate-value-activation-value-usage-analysis)
    - [gpu memory usage test](#gpu-memory-usage-test)
      - [batch\_size=1](#batch_size1)
      - [batch\_size=2](#batch_size2)
- [Time and space test](#time-and-space-test)

# Space occupancy analysis

## Parameter quantity and model size analysis and testing

### Parameter quantity and model size analysis

#### Hyperparameters

| Hyperparameter name   | Meaning                           | Abbreviation |
|-----------------------|-----------------------------------|--------------|
| vocabulary_size       | vocabulary size                   | V            |
| embed_dim             | dimension of word embedding layer | E            |
| num_transformer_block | number of transformer layers      | N            |
| batch_size            | number of samples in a batch      | B            |
| seq_len               | sequence length                   | S            |

#### llama2 model structure

* RoPE matrix
    * $max\_position * embedding\_size$
* word_embedding
    * $src\_vocab\_size * embedding\_dim = V * E$
* N * transformer_block
    * attention
        * $W^Q,W^K,W^V,W^O$
            * $4 * embedding\_dim^2 = 4E^2$
        * attention_norm
            * $embedding\_dim = E$
    * feed-forward
        * $W_1,W_2,W_3$ (dim,hidden_dim)
            * $3 * embedding\_dim * (4 * embedding\_dim) = 12 * E^2$
        * ffn_norm
            * $embedding\_dim = E$
* norm
    * weight
        * $embedding\_dim = E$
* lm_head
    * $tgt\_vocab\_size * embedding\_dim = V * E$

$$total\_params= VE + N  (4 E^2 + E + 12 E^2 + E) + E + V E$$

#### llama_7B parameter number

For example, let

* E = 4096
* N = 32
* V = 32000

Calculated total number of parameters$ = 8,852,344,832 = 8.8 * 10 ^ 9 \approx 16.4GB (bf16) $
if let 4=8/3, Calculated total number of parameters$ = 6,704,861,184 = 6.7 * 10 ^ 9 \approx 12.48GB (bf16) $

### Parameter number and model size test

```
before initialize_text_to_text_model===================================
GPU Allocated Memory: 0.00 GB
GPU max_memory_allocated 0.0 GB
CPU memory size of all:1008GB
CPU memory used:99GB(10%)
CPU memory available :71GB
after initialize_text_to_text_model finished===========================
GPU Allocated Memory: 0.00 GB
GPU max_memory_allocated 0.0 GB
CPU memory size of all:1008GB
CPU memory used:112GB(12%)
CPU memory available :58GB
=======================================================================
Total parameters6738415616 = 6.738415616*10^9
theoretical model size = 12.551277160644531GB
```

## gpu memory usage analysis and test

### gpu memory usage analysis

#### Model usage analysis

$$Model\ parameter\ quantity = VE + N (4 E^2 + E + 12 E^2 + E) + E + VE$$
Take llama7B=14GB

#### Intermediate value (activation value) usage analysis

transformer_block:attention structure

```plantuml
(X)-->(Q')
(W^Q)-->(Q')
(Q')-->(Q):RoPE
(X)-->(K')
(W^K)-->(K')
(K')-->(K):RoPE
(X)-->(V)
(W^V)-->(V)
(Q)-->(QK^T)
(K)-->(QK^T)
(QK^T)-->(A):softmax(QK^T/d))
(A)-->(AV)
(V)-->(AV)
(AV)-->(O)
(W^O)-->(O)
```

transformer_block:feed-forward structure

```plantuml
(X)-->(XW_1)
(W_1)-->(XW_1)
(X)-->(XW_2)
(W_2)-->(XW_2)
(XW_1)-->(H)
(XW_2)-->(H)
(H)-->(silu)
(W_3)-->(siluW_3)
(silu)-->(siluW_3)
```

Additional intermediate tensors saved for back propagation = Q, Q', K, K', (QK^T), A, V, (AV), (O), (XW_1), (XW_2), H,
silu, siluW
$$
Number\ of\ additional\ intermediate\ parameters\ saved\ for\ back\ propagation = 8SE + 2S^2 + 4S*4E \approx 24SE
$$
If the sequence length is 400
$$Activation/model = 24SE/16E^2 = 24S/16E=20*400/16*4096=0.146$$
$$0.146*14GB=2.05GB$$

### gpu memory usage test

| batch_size\ stage | before forward | after forward(before backward) | after backward | saved tenor |
|-------------------|----------------|--------------------------------|----------------|-------------|
| 1                 | 12.61GB(0GB)   | 13.66GB(0GB)                   | 12.98GB(13GB)  | 1.05GB      |
| 2                 | 12.61GB(0GB)   | 16.33GN(0GB)                   | 13.23GB(12GB)  | 3.71GB      |
| 3                 | 12.61GB(0GB)   | 18.21GB(0GB)                   | 13.41GB(12GB)  | 5.6GB       |

The brackets are memory usage

#### batch_size=1

````
before forward===========================================================
GPU Allocated Memory: 12.61 GB
GPU max_memory_allocated 12.613796710968018 GB
CPU memory size of all:1008GB
CPU memory used:95GB(10%)
CPU memory available :74GB
before backward===========================================
GPU Allocated Memory: 13.66 GB
GPU max_memory_allocated 13.657839298248291 GB
CPU memory size of all:1008GB
CPU memory used:95GB(10%)
CPU memory available :74GB
after backward ===========================================================
GPU Allocated Memory: 12.98 GB
GPU max_memory_allocated 13.900128841400146 GB
CPU memory size of all:1008GB
CPU memory used:109GB(12%)
CPU memory available :61GB
after clear gradient======================================================
GPU Allocated Memory: 12.74 GB
GPU max_memory_allocated 13.900128841400146 GB
CPU memory size of all:1008GB
CPU memory used:109GB(12%)
CPU memory available :61GB

````

#### batch_size=2

```
before forward===========================================================
GPU Allocated Memory: 12.61 GB
GPU max_memory_allocated 12.61380672454834 GB
CPU memory size of all:1008GB
CPU memory used:98GB(10%)
CPU memory available :71GB
before backward===========================================
GPU Allocated Memory: 16.33 GB
GPU max_memory_allocated 16.397658824920654 GB
CPU memory size of all:1008GB
CPU memory used:98GB(10%)
CPU memory available :70GB
after backward ===========================================================
GPU Allocated Memory: 13.23 GB
GPU max_memory_allocated 16.548716068267822 GB
CPU memory size of all:1008GB
CPU memory used:111GB(12%)
CPU memory available :58GB
after clear gradient======================================================
GPU Allocated Memory: 12.99 GB
GPU max_memory_allocated 16.548716068267822 GB
CPU memory size of all:1008GB
CPU memory used:111GB(12%)
CPU memory available :58GB
```
