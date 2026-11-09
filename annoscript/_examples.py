"""Portable example data; this module has no runtime/backend dependencies."""

from dataclasses import dataclass

from annoscript._data import freeze, snapshot, thaw


def example_data(state, answer):
    """The closed ``example`` intrinsic: return detached, finite JSON data."""
    return {"state": snapshot(state, state=True), "answer": snapshot(answer)}


@dataclass(frozen=True)
class Example:
    """An input and its task answer, using the enclosing operation's parameters.

    Answers contain only the business value, never confidence or probabilities.
    Nested values are immutable snapshots. Compiled examples carry ordinary JSON,
    not Python objects. Task/criteria overrides are intentionally unsupported.
    """

    state: object
    answer: object

    def __post_init__(self):
        data = example_data(self.state, self.answer)
        object.__setattr__(self, "state", freeze(data["state"]))
        object.__setattr__(self, "answer", freeze(data["answer"]))

    def to_dict(self):
        return {"state": thaw(self.state), "answer": thaw(self.answer)}

    def __deepcopy__(self, memo):
        return self
