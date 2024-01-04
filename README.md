# Mistral MindSpore version

This repository contains mindspore implemented code to run mistral 7B model.

Blog: [https://mistral.ai/news/announcing-mistral-7b/](https://mistral.ai/news/announcing-mistral-7b/)\
Discord: [https://discord.com/invite/mistralai](https://discord.com/invite/mistralai)\
Documentation: [https://docs.mistral.ai/](https://docs.mistral.ai/)\
Guardrailing: [https://docs.mistral.ai/usage/guardrailing](https://docs.mistral.ai/usage/guardrailing)


## Installation

```bash
pip install -r requirements.txt
```

## Download the model
```bash
wget https://files.mistral-7b-v0-1.mistral.ai/mistral-7B-v0.1.tar
tar -xf mistral-7B-v0.1.tar
```

## Convert checkpoint

```bash
python convert.py --pth_file /path/to/mistral-7B-v0.1/consolidated.00.pth
```

## Run the model

```bash
python -m main demo /path/to/mistral-7B-v0.1/
# To give your own prompts
python -m main interactive /path/to/mistral-7B-v0.1/
```
Change `temperature` or `max_tokens` using:
```bash
python -m main interactive /path/to/mistral-7B-v0.1/ --max_tokens 256 --temperature 1.0
```

### Running large models

When running models that are too large to fit a single GPU's memory, use pipeline parallelism (PP) and `mpirun`. This is needed to run `Mixtral-7B-8x`. The code below does 2-way PP.

```bash
wget https://files.mixtral-8x7b-v0-1.mistral.ai/Mixtral-8x7B-v0.1-Instruct.tar
tar -xf Mixtral-8x7B-v0.1-Instruct.tar
```

```bash
python convert.py --pth_file /path/to/Mixtral-8x7B-v0.1-Instruct/consolidated.00.pth
```

```bash
mpirun -n 2 python -m main demo /path/to/Mixtral-8x7B-v0.1-Instruct/ --num_pipeline_ranks=2
```

> [!Note]
> PP is not supported when running in interactive mode.

# Sliding window attention

## Vanilla attention

Attention is how information is shared between tokens in a sequence.
In vanilla transformers, attention follows a causal mask: each token in the sequence can attend to itself and all the tokens in the past.
This ensures that the model is causal, i.e. it can only use information from the past to predict the future.


![Causal attention mask](assets/full_attention.png)

## Sliding window to speed-up inference and reduce memory pressure

The number of operations of attention is quadratic in the sequence length, and the memory pressure is linear in the sequence length.
At inference time, this incurs higher latency and smaller throughput due to reduced cache availability.
To alleviate this issue, we use a sliding window attention [1,2]: each token can attend to at most W tokens in the past (here, W=3).

![Sliding window attention](assets/sliding_attention.png)

Note that tokens outside the sliding window still influence next word prediction. 
At each attention layer, information can move forward by W tokens at most: after two attention layers, information can move forward by 2W tokens, etc.
For instance in a sequence of length 16K and a sliding window of 4K, after 4 layers, information has propagated to the full sequence length.

![Attention through layers](assets/attention_through_layers.png)

Empirically, we see that longer contexts do help *even outside the sliding window* but when the sequence length becomes too large, the model does not use the full context anymore.

## Rolling buffer cache

We implement a rolling buffer cache.
The cache has a fixed size of W, and we store the (key, value) for position i in cache position i % W.
When the position i is larger than W, past values in the cache are overwritten.

![Rolling cache](assets/rolling_cache.png)

## Pre-fill and chunking

When generating a sequence, we need to predict tokens one-by-one, as each token is conditioned on the previous ones.
However, the prompt is known in advance, and we can pre-fill the (k, v) cache with the prompt.
If the prompt is very large, we can chunk it into smaller pieces, and pre-fill the cache with each chunk.
For this we can choose as chunk size the window size. For each chunk, we thus need to compute the attention over the cache and over the chunk.

![Chunking](assets/chunking.png)


# Sparse Mixture of Experts (SMoE)

Sparse Mixture of Experts allows one to decouple throughput from memory costs by only activating subsets of the overall model for each token. In this approach, each token is assigned to one or more "experts" -- a separate set of weights -- and only processed by sunch experts. This division happens at feedforward layers of the model. The expert models specialize in different aspects of the data, allowing them to capture complex patterns and make more accurate predictions.

![SMoE](assets/smoe.png)

## Pipeline Parallelism

Pipeline parallelism is a set of techniques for partitioning models, enabling the distribution of a large model across multiple GPUs. We provide a simple implementation of pipeline parallelism, which allows our larger models to be executed within the memory constraints of modern GPUs. Note that this implementation favours simplicity over throughput efficiency, and most notabably does not include microbatching.


## References

[1] [Generating Long Sequences with Sparse Transformers, Child et al. 2019](https://arxiv.org/pdf/1904.10509.pdf)

[2] [Longformer: The Long-Document Transformer, Beltagy et al. 2020](https://arxiv.org/pdf/2004.05150v2.pdf)