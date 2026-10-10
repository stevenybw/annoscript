from abc import ABC, abstractmethod
from typing import Any, List, Dict
from dataclasses import dataclass


@dataclass
class NoulResponse:
    noul: float


@dataclass
class ChoiceResponse:
    choice: str
    confidence: float
    probabilities: Dict[str, float]


@dataclass
class ScoreResponse:
    probabilities: Dict[int, float]
    score: float
    legend: Dict[int, str]
    confidence: float


class Lev(ABC):
    """Decision interface whose questions share a state and model."""

    def __init__(self, state: str | Dict[str, Any] | List[Any], model: str):
        self.state = state
        self.model = model

    @abstractmethod
    def noul(
        self,
        instructions: str,
        criteria_false: str | None,
        criteria_true: str | None,
    ) -> NoulResponse:
        """Evaluate a yes/no question using this instance's state and model."""
        raise NotImplementedError

    @abstractmethod
    def choice(self, instructions: str, criteria: Dict[str, str]) -> ChoiceResponse:
        """Select an option using this instance's state and model."""
        raise NotImplementedError

    @abstractmethod
    def score(self, instructions: str, criteria: List[str]) -> ScoreResponse:
        """Score the state against ordered criteria using this instance's model."""
        raise NotImplementedError
