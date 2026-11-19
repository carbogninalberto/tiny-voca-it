"""Faster Qwen3-TTS generation: replace the nested Hugging Face ``generate()`` of the code predictor
(called once per codec frame, 12 times per second of audio, to sample 15 sub-codes) with a lean loop.

The per-frame ``generate()`` sets up logits processors, stopping criteria and a new cache every time;
that Python overhead, not GPU work, bounds generation (GPU ~15% busy). The loop below runs the same
forward passes with one DynamicCache and samples with the same rule as HF (temperature, then top-k,
then top-p, multinomial), so the output distribution is unchanged.

    import qwen_fast; qwen_fast.patch()      # before generating (enabled by QWEN_FAST=1 in the scripts)
"""
from types import SimpleNamespace

import torch


def _sample(logits, do_sample, top_k, top_p, temperature):
    if not do_sample:
        return logits.argmax(-1)
    if temperature and temperature != 1.0:
        logits = logits / temperature
    if top_k and top_k > 0:
        kth = torch.topk(logits, min(top_k, logits.shape[-1]), dim=-1).values[..., -1, None]
        logits = logits.masked_fill(logits < kth, float("-inf"))
    if top_p is not None and top_p < 1.0:
        ordered, index = torch.sort(logits, descending=True)
        cumulative = ordered.softmax(-1).cumsum(-1)
        remove = cumulative - ordered.softmax(-1) > top_p
        logits = logits.masked_fill(remove.scatter(-1, index, remove), float("-inf"))
    return torch.multinomial(logits.softmax(-1), 1)[:, 0]


def fast_generate(self, inputs_embeds=None, max_new_tokens=15, do_sample=True, top_p=1.0, top_k=50, temperature=0.9, **_):
    from transformers.cache_utils import DynamicCache
    defaults = self.generation_config
    do_sample = defaults.do_sample if do_sample is None else do_sample
    top_k = defaults.top_k if top_k is None else top_k
    top_p = defaults.top_p if top_p is None else top_p
    temperature = defaults.temperature if temperature is None else temperature
    cache = DynamicCache()
    out = self(inputs_embeds=inputs_embeds, past_key_values=cache, use_cache=True)
    steps, tokens = out.generation_steps, []
    for i in range(max_new_tokens):
        token = _sample(out.logits[:, -1, :].float(), do_sample, top_k, top_p, temperature)
        tokens.append(token)
        if i + 1 == max_new_tokens:
            break
        out = self(input_ids=token[:, None], past_key_values=cache, use_cache=True, generation_steps=steps)
        steps = out.generation_steps
    return SimpleNamespace(sequences=torch.stack(tokens, 1))


def patch():
    from qwen_tts.core.models import modeling_qwen3_tts as m
    m.Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.generate = torch.inference_mode()(fast_generate)


class _Graphed:
    """The whole 15-step sub-code loop of one batch size captured as a CUDA graph (static KV cache,
    explicit cache positions, sampling inside the graph). Replaying it costs one launch per frame
    instead of a few hundred."""

    def __init__(self, predictor, batch, width, device, dtype, steps, sampling):
        from transformers.cache_utils import StaticCache
        self.input = torch.zeros(batch, 2, width, device=device, dtype=dtype)
        self.cache = StaticCache(config=predictor.config, max_cache_len=steps + 2)
        positions = [torch.arange(0, 2, device=device)] + [torch.tensor([2 + i], device=device) for i in range(steps - 1)]

        def loop():
            out = predictor(inputs_embeds=self.input, past_key_values=self.cache, use_cache=True, cache_position=positions[0])
            step, tokens = out.generation_steps, []
            for i in range(steps):
                token = _sample(out.logits[:, -1, :].float(), *sampling)
                tokens.append(token)
                if i + 1 == steps:
                    break
                out = predictor(input_ids=token[:, None], past_key_values=self.cache, use_cache=True,
                                generation_steps=step, cache_position=positions[i + 1])
                step = out.generation_steps
            return torch.stack(tokens, 1)

        stream = torch.cuda.Stream()
        stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(stream):
            for _ in range(2):
                loop()
        torch.cuda.current_stream().wait_stream(stream)
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = loop()

    def __call__(self, inputs_embeds):
        self.input.copy_(inputs_embeds)
        self.graph.replay()
        return self.output.clone()


def graphed_generate(self, inputs_embeds=None, max_new_tokens=15, do_sample=True, top_p=1.0, top_k=50, temperature=0.9, **_):
    defaults = self.generation_config
    sampling = (defaults.do_sample if do_sample is None else do_sample, defaults.top_k if top_k is None else top_k,
                defaults.top_p if top_p is None else top_p, defaults.temperature if temperature is None else temperature)
    key = (inputs_embeds.shape[0], inputs_embeds.shape[1], max_new_tokens, sampling)
    graphs = self.__dict__.setdefault("_graphs", {})
    if inputs_embeds.shape[1] != 2 or not inputs_embeds.is_cuda:
        return fast_generate(self, inputs_embeds, max_new_tokens, *sampling)
    if key not in graphs:
        graphs[key] = _Graphed(self, inputs_embeds.shape[0], inputs_embeds.shape[2], inputs_embeds.device, inputs_embeds.dtype, max_new_tokens, sampling)
    return SimpleNamespace(sequences=graphs[key](inputs_embeds))


def patch_graphs():
    from qwen_tts.core.models import modeling_qwen3_tts as m
    m.Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.generate = torch.inference_mode()(graphed_generate)
