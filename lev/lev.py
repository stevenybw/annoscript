from abc import ABC, abstractmethod
from typing import Any, List, Dict
from dataclasses import dataclass


def constrain(condition: bool, *, message: str | None = None) -> None:
    """Declare a hard constraint in a compiled Lev eDSL program.

    The compiler captures the pure ``condition`` expression and its dependencies
    before Python evaluates the arguments. It may propagate the constraint to
    related inference operations whose results are still adjustable within the
    current solving scope. A constraint inside a branch applies only on that
    execution path.

    Successful execution past this call requires the condition to hold, as
    established by proof or runtime validation. If a satisfying result cannot
    be obtained, execution must report failure; retry and repair budgets are
    execution-policy concerns. ``message`` optionally describes the constraint
    for failure diagnostics. Constraints remain active in optimized execution.

    This function is an eDSL intrinsic, not a standalone solver or validator.
    Direct Python calls always raise ``RuntimeError``, even for a true condition:
    the evaluated argument no longer carries the expression needed to propagate
    constraints. The eDSL compiler must lower this call to a constraint operation.
    """
    raise RuntimeError(
        "constrain() requires a compiled Lev eDSL program; "
        "direct Python execution cannot propagate constraints."
    )


@dataclass
class NoulResponse:
    noul: float


@dataclass
class ChoiceResponse:
    """Choice result; the provider guarantees finite confidence in [0, 1]."""

    choice: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class ScoreResponse:
    probabilities: Dict[int, float]
    score: float
    legend: Dict[int, str]
    confidence: float


@dataclass
class ExtractResponse:
    items: List[str]


@dataclass
class GenerateResponse:
    text: str


@dataclass
class GenerateListResponse:
    items: List[str]


class Lev(ABC):
    """Decision, extraction, and generation operations sharing execution context.

    The runtime supplies the program's system_prompt when constructing the
    per-record instance. Implementations must apply it to every model operation,
    including operations called from helpers, alongside the per-call task
    instructions. It does not override API guarantees or hard constraints.
    """

    def __init__(
        self,
        state: str | Dict[str, Any] | List[Any],
        model: str,
        *,
        system_prompt: str | None = None,
    ):
        self.state = state
        self.model = model
        self.system_prompt = system_prompt

    @abstractmethod
    def noul(
        self,
        instructions: str,
        criteria_false: str | None = None,
        criteria_true: str | None = None,
    ) -> NoulResponse:
        """Evaluate a yes/no question using this instance's state and model."""
        raise NotImplementedError

    @abstractmethod
    def choice(self, instructions: str, criteria: Dict[str, str]) -> ChoiceResponse:
        """Select an option using this instance's state and model.

        A successful result must have ``result.choice in criteria`` and a finite
        ``result.confidence`` in the inclusive range [0, 1], excluding NaN and
        infinities. Implementations must enforce these guarantees and report
        failure if a valid result cannot be produced.
        """
        raise NotImplementedError

    @abstractmethod
    def score(self, instructions: str, criteria: List[str]) -> ScoreResponse:
        """Score the state against ordered criteria using this instance's model."""
        raise NotImplementedError

    @abstractmethod
    def extract(
        self,
        instructions: str,
        *,
        source: str | None = None,
        min_items: int = 0,
        max_items: int,
        max_length_per_item: int,
    ) -> ExtractResponse:
        """Extract verbatim passages using this instance's state and model.

        ``source`` is the only text from which items may be extracted. If it is
        omitted, use ``self.state`` when the state is a string; structured states
        require an explicit source, such as ``source=self.state["text"]``.
        Other state fields may provide context, but are not extraction sources.

        Each item must be a nonempty, contiguous substring of the source, without
        rewriting, normalization, or concatenation of separate passages. The
        number of items must satisfy the inclusive ``min_items`` / ``max_items``
        bounds. Required ``max_length_per_item`` is a suggested Unicode code-point
        length, measured by Python's ``len``, and contributes to the token budget.
        Exceeding the suggestion does not invalidate an otherwise valid item.
        Return an empty list when nothing matches and ``min_items`` is zero.

        Implementations must raise ``ValueError`` for a missing source with a
        structured state, negative item bounds, an inverted item range, or a
        nonpositive ``max_length_per_item``. If extraction cannot satisfy the
        constraints, report failure rather than inventing items or returning a
        successful result that violates the bounds. Matching the source does
        not by itself establish that an item semantically supports a claim.
        """
        raise NotImplementedError

    @abstractmethod
    def generate(
        self,
        instructions: str,
        *,
        min_length: int = 0,
        max_length: int,
        max_num_tokens: int | None = None,
    ) -> GenerateResponse:
        """Generate text using this instance's state and model.

        The result is always a string, never ``None``, and may paraphrase or
        synthesize the state as directed by ``instructions``. ``min_length`` and
        required ``max_length`` are suggested Unicode code-point lengths, measured by
        Python's ``len``. A positive ``min_length`` still requires nonempty text,
        but other deviations from the suggested range are not failures.

        ``max_num_tokens`` is an independent output limit for a standalone
        generation. When Runtime batches ready calls, it contributes to a shared
        budget and is not an enforced per-answer token limit. ``None`` uses the
        client's assumed_avg_num_tokens_per_char (default 2) times max_length,
        rounded up with a minimum of one token. Suggested lengths are not hard bounds.
        Exhausting the budget must not silently turn an incomplete generation
        into a successful response.

        Implementations must raise ``ValueError`` for negative length bounds,
        an inverted length range, or a nonpositive ``max_num_tokens``. If valid
        output cannot be produced, report failure rather than returning a
        successful result that violates the constraints. These constraints do
        not guarantee factual accuracy. Branching and recovery remain the
        caller's responsibility.
        """
        raise NotImplementedError

    @abstractmethod
    def generate_list(
        self,
        instructions: str,
        *,
        min_items: int = 0,
        max_items: int,
        min_length_per_item: int = 0,
        max_length_per_item: int,
        max_num_tokens: int | None = None,
    ) -> GenerateListResponse:
        """Generate a list of texts using this instance's state and model.

        The result is always a list of strings, with no ``None`` items. Each
        item may paraphrase or synthesize the state as directed by
        ``instructions``. The item count must satisfy the inclusive
        ``min_items`` / ``max_items`` bounds; an empty list is allowed when
        ``min_items=0``.

        ``min_length_per_item`` and required ``max_length_per_item`` are suggested
        Unicode code-point lengths for each string, measured by Python's ``len``.
        A positive minimum still requires nonempty items. Required ``max_items``
        is a hard count limit. Neither required maximum accepts None.

        ``max_num_tokens`` is a generation budget for the entire list,
        including formatting, in the model's tokens. In a Runtime batch it
        contributes to a shared budget, without an independent per-list token
        limit. ``None`` estimates ceil(assumed_avg_num_tokens_per_char * max_items *
        max_length_per_item) + 32 + 8 * max_items. Extraction uses the same
        formula. Item count and nonempty requirements remain strict. Exhausting the budget must not silently
        turn an incomplete list or item into a successful response.

        Implementations must raise ``ValueError`` for negative item or length
        bounds, inverted ranges, or a nonpositive ``max_num_tokens``. If valid
        output cannot be produced, report failure rather than returning a
        successful result that violates the constraints. These constraints do
        not guarantee factual accuracy. Branching and recovery remain the
        caller's responsibility.
        """
        raise NotImplementedError
