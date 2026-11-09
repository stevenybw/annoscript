"""Conservative, deterministic optimization of structured AnnoScript IR.

The input is never mutated and no source, model, or user function is executed.
Passes preserve ordinary execution and constraint bindings; they neither solve
constraints nor treat an asserted condition as a fact. Unknown effects remain.
Individual passes are implementation details; optimize_program is the public API.
"""

from annoscript.ir import Program
from .control_flow import simplify_control_flow
from .dead_assignments import eliminate_dead_assignments
from .folding import fold_constants
from .propagation import propagate_values
from .temporaries import merge_temporaries

__all__ = ["optimize_program"]

_PASSES = (
    propagate_values, fold_constants, simplify_control_flow,
    eliminate_dead_assignments, merge_temporaries,
)


def optimize_program(program: Program) -> Program:
    """Optimize a Program to a fixed point without changing the IR v1 schema.

The passes only propagate atoms, fold bounded scalars, or remove structure;
they do not expand loops, duplicate statements, or introduce new variables.
JSON equality is intentional: dataclass equality conflates True, 1 and 1.0.
"""
    if not isinstance(program, Program):
        raise TypeError("optimize_program expects an ir.Program")
    previous = program.to_json(indent=None)
    while True:
        for transform in _PASSES:
            program = transform(program)
        current = program.to_json(indent=None)
        if current == previous:
            return program
        previous = current
