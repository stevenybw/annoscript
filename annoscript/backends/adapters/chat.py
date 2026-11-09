"""generation_chat_v1: existing messages, examples and protocol-specific envelopes."""

from .._prompt import messages


class ChatGenerationAdapter:
    def single_payload(self, context, call, *, model, profile, disable_thinking,
                       max_tokens, grammar, sampling=None, repair_feedback=None):
        payload = {"model": model,
                   "messages": messages(context, [call], repair_feedback=repair_feedback)}
        if disable_thinking:
            if profile == "qwen_vllm":
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            else:
                payload["thinking"] = {"type": "disabled"}
        payload["max_tokens"] = max_tokens
        if profile == "qwen_vllm":
            payload["structured_outputs"] = {"grammar": grammar([call])}
        elif call.operation != "generate":
            payload["response_format"] = {"type": "json_object"}
        if sampling is not None:
            payload.update(sampling)
        return payload

    def batch_payload(self, context, calls, *, model, profile, disable_thinking,
                      max_tokens, grammar, sampling=None, rule_dedup=False):
        payload = {"model": model, "max_tokens": max_tokens,
                   "messages": messages(context, calls, batch=True, rule_dedup=rule_dedup)}
        if profile == "qwen_vllm":
            payload["structured_outputs"] = {"grammar": grammar(calls, batch=True)}
            if disable_thinking:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
        else:
            payload["response_format"] = {"type": "json_object"}
            if disable_thinking:
                payload["thinking"] = {"type": "disabled"}
        if sampling is not None:
            payload.update(sampling)
        return payload
