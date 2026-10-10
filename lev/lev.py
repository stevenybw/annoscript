from typing import List, Dict
from dataclasses import dataclass

@dataclass
class NoulResponse:
    noul: float

def noul(instructions: str, criteria_false: str, criteria_true: str) -> NoulResponse:
    pass


@dataclass
class ChoiceResponse:
    choice: str
    confidence: float
    probabilities: Dict[str, float]


def choice(instructions: str, criteria: List[Dict[str, str]]) -> ChoiceResponse:
    pass


@dataclass
class ScoreResponse:
    probabilities: Dict[int, float]
    score: float
    legend: Dict[int, str]
    confidence: float


def score(instructions: str, criteria: List[str]) -> ScoreResponse:
    pass
