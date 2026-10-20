"""Request-local example sharing; never modifies the current inference state."""

import json


EXAMPLE_CONTRACT = (
    "examples 是任务的输入与正确答案示范，只用于理解规则，不是当前 state 的事实。"
    "answer 是该算子的值，不包含置信度；当前输出仍遵守指定格式。"
)
EXAMPLE_REFERENCES = "示例的 state_ref 引用 example_states 中的输入。"


def share_examples(tasks):
    """Factor detached task arguments, preserving example and task order."""
    states, identities = {}, {}
    for task in tasks:
        for example in task.get("examples", []):
            state = example.pop("state")
            key = json.dumps(state, ensure_ascii=False, allow_nan=False,
                             sort_keys=True, separators=(",", ":"))
            if key not in identities:
                identity = f"e{len(states)}"
                identities[key] = identity
                states[identity] = state
            example["state_ref"] = identities[key]
    return states
