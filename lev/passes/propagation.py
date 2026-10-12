"""Forward propagation of scalar constants and copies of existing values."""

from dataclasses import replace

from lev import ir
from ._analysis import (
    block_writes, bound_before, definitions, invalidate, join_exits,
    map_expressions, resolve, rewrite_expr,
)


def propagate_values(program: ir.Program) -> ir.Program:
    points = bound_before(program)

    def block(body, incoming):
        facts = dict(incoming)
        output = []
        falls = True
        for original in body:
            statement = map_expressions(original, lambda e: rewrite_expr(e, lambda v: resolve(v, facts)))
            if isinstance(statement, ir.If):
                yes, a, af = block(statement.then_body, facts)
                no, b, bf = block(statement.else_body, facts)
                facts = join_exits(a, af, b, bf)
                statement = replace(statement, then_body=yes, else_body=no)
                falls = af or bf
            elif isinstance(statement, ir.For):
                writes = definitions(statement) | block_writes(statement.body)
                facts = invalidate(facts, writes)
                nested, _, _ = block(statement.body, facts)
                statement = replace(statement, body=nested)
            elif isinstance(statement, ir.Assign):
                value = statement.value
                if not (isinstance(value, ir.Var) and value.name == statement.target):
                    facts = invalidate(facts, {statement.target})
                    if isinstance(value, ir.Literal) or (
                        isinstance(value, ir.Var) and value.name in points.get(statement.id, ())
                    ):
                        facts[statement.target] = value
            elif isinstance(statement, ir.Constrain):
                facts.clear()
            elif isinstance(statement, ir.Return):
                falls = False
            else:
                facts = invalidate(facts, definitions(statement))
            output.append(statement)
            if not falls:
                # Unreachable code is removed by the control-flow pass.
                output.extend(body[len(output):])
                break
        return tuple(output), facts, falls

    return replace(program, body=block(program.body, {})[0])
