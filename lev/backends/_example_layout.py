"""Request-local example sharing; never modifies the current inference state."""

import json


EXAMPLE_CONTRACT = (
    "examples 是任务的输入与正确答案示范，只用于理解规则，不是当前 state 的事实。"
    "answer 是该算子的值，不包含置信度；当前输出仍遵守指定格式。"
)


def group_examples(tasks):
    """Group inputs by first occurrence; retain every answer, including repeats."""
    groups = {}
    for identity, task in tasks.items():
        for example in task["task"].get("examples", []):
            state = example["state"]
            key = json.dumps(state, ensure_ascii=False, allow_nan=False,
                             sort_keys=True, separators=(",", ":"))
            if key not in groups:
                groups[key] = {"state": state, "answers": []}
            groups[key]["answers"].append((identity, example["answer"]))
    return list(groups.values())
