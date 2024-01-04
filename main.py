import logging
import fire
from typing import List
from pathlib import Path

import mindspore
from mindspore import ops
from mindspore.communication import init, get_rank

from mistral.model import Transformer
from mistral.tokenizer import Tokenizer
from mistral.cache import RotatingBufferCache

# mindspore.set_context(pynative_synchronize=True)

def sample_top_p(probs: mindspore.Tensor, p: float):
    assert 0 <= p <= 1

    probs_sort, probs_idx = ops.sort(probs, axis=-1, descending=True)
    probs_sum = ops.cumsum(probs_sort, axis=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(ops=-1, keepdim=True))
    next_token = ops.multinomial(probs_sort, num_samples=1)
    return ops.gather_elements(probs_idx, -1, next_token)


def sample(logits: mindspore.Tensor, temperature: float, top_p: float):
    if temperature > 0:
        probs = ops.softmax(logits / temperature, axis=-1)
        next_token = sample_top_p(probs, top_p)
    else:
        next_token = ops.argmax(logits, dim=-1).unsqueeze(0)

    return next_token.reshape(-1)


def generate(prompts: List[str], model: Transformer, tokenizer: Tokenizer, *, max_tokens: int,  temperature: float, chunk_size: int = None):
    model = model.set_train(False)
    B, V = len(prompts), model.args.vocab_size

    # Tokenize
    encoded_prompts = [tokenizer.encode(prompt, bos=True) for prompt in prompts]
    seqlens = [len(x) for x in encoded_prompts]

    # Cache
    cache_window = max(seqlens) + max_tokens
    if model.args.sliding_window is not None and cache_window > model.args.sliding_window:
        cache_window = model.args.sliding_window
    cache = RotatingBufferCache(
        model.n_local_layers,
        model.args.max_batch_size,
        cache_window,
        model.args.n_kv_heads,
        model.args.head_dim,
    )
    cache = cache.to(dtype=model.dtype)
    cache.reset()
    
    # Bookkeeping
    logprobs = [[] for _ in range(B)]
    last_token_prelogits = None

    # One chunk if size not specified
    max_prompt_len = max(seqlens)
    if chunk_size is None:
        chunk_size = max_prompt_len

    # Encode prompt by chunks
    for s in range(0, max_prompt_len, chunk_size):
        prompt_chunks = [p[s:s+chunk_size] for p in encoded_prompts]
        assert all(len(p) > 0 for p in prompt_chunks)
        prelogits = model(
            mindspore.Tensor(sum(prompt_chunks, []), dtype=mindspore.int64),
            seqlens=[len(p) for p in prompt_chunks],
            cache=cache
        )
        logits = ops.log_softmax(prelogits, axis=-1)

        if last_token_prelogits is not None:
            # Pass > 1
            last_token_logits = ops.log_softmax(last_token_prelogits, axis=-1)
            for i_seq in range(B):
                logprobs[i_seq].append(last_token_logits[i_seq, prompt_chunks[i_seq][0]].item())

        offset = 0
        for i_seq, sequence in enumerate(prompt_chunks):
            logprobs[i_seq].extend([logits[offset + i, sequence[i + 1]].item() for i in range(len(sequence) - 1)])
            offset += len(sequence)

        last_token_prelogits = prelogits.index_select(0, mindspore.Tensor([len(p) for p in prompt_chunks]).cumsum(axis=0) - 1)
        assert last_token_prelogits.shape == (B, V)

    # decode
    generated_tokens = []
    assert last_token_prelogits is not None
    for i_token in range(max_tokens):
        next_token = sample(last_token_prelogits, temperature=temperature, top_p=0.8)
        last_token_logits = ops.log_softmax(last_token_prelogits, axis=-1)
        for i in range(B):
            logprobs[i].append(last_token_logits[i, next_token[i]].item())

        generated_tokens.append(next_token[:, None])
        last_token_prelogits = model(next_token, seqlens=[1] * len(prompts), cache=cache)
        assert last_token_prelogits.shape == (B, V)
    generated_words = []
    if generated_tokens:
        generated_tokens = ops.cat(generated_tokens, 1)
        for i, x in enumerate(encoded_prompts):
            generated_words.append(tokenizer.decode(x + generated_tokens[i].tolist()))
    return generated_words, logprobs


def interactive(model_path: str, max_tokens: int = 35, temperature: float = 0.7, instruct: bool = False):
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(Path(model_path), max_batch_size=3)

    while True:
        prompt = input("Prompt: ")
        if instruct:
            prompt = f"[INST] {prompt} [/INST]"
        res, _logprobs = generate(
            [prompt],
            transformer,
            tokenizer,
            max_tokens=max_tokens,
            temperature=temperature,
        )
        print(res[0])
        print("=====================")


def demo(
    model_path: str, max_tokens: int = 35, temperature: float = 0, num_pipeline_ranks=1
):
    if num_pipeline_ranks > 1:
        backend_name = "nccl" if mindspore.get_context('device_target') == 'GPU' else 'hccl'
        init(backend_name)
        should_print = get_rank() == 0
    else:
        should_print = True
    tokenizer = Tokenizer(str(Path(model_path) / "tokenizer.model"))
    transformer = Transformer.from_folder(
        Path(model_path), max_batch_size=3, num_pipeline_ranks=num_pipeline_ranks
    )

    res, _logprobs = generate(
        [
            "This is a test",
            "This is another great test",
            "This is a third test, mistral AI is very good at testing. ",
        ],
        transformer,
        tokenizer,
        max_tokens=max_tokens,
        temperature=temperature,
    )
    if should_print:
        for x,l in zip(res, _logprobs):
            print(x)
            logging.debug('Logprobs: %s',l)
            print("=====================")

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    fire.Fire({
        "interactive": interactive,
        "demo": demo,
    })