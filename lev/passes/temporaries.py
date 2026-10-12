"""Fuse a single-use assignment into the next whole-expression consumer.

No computation crosses another statement or enters a conditional subexpression.
Global static use/definition counts deliberately reject ambiguous bindings.
"""

from collections import Counter
from dataclasses import replace

from lev import ir
from ._analysis import definitions, direct_reads, walk_statements


def merge_temporaries(program: ir.Program) -> ir.Program:
    uses, writes = Counter(), Counter()
    for statement in walk_statements(program.body):
        uses.update(direct_reads(statement))
        writes.update(definitions(statement))

    def block(body):
        output = []
        for statement in body:
            if isinstance(statement, ir.If):
                statement = replace(statement, then_body=block(statement.then_body), else_body=block(statement.else_body))
            elif isinstance(statement, ir.For):
                statement = replace(statement, body=block(statement.body))
            field = None
            if isinstance(statement, (ir.Assign, ir.Return)):
                field = "value"
            elif isinstance(statement, ir.If):
                field = "condition"
            elif isinstance(statement, ir.For):
                field = "iterable"
            while field and output and isinstance(output[-1], ir.Assign):
                previous = output[-1]
                value = getattr(statement, field)
                if not (isinstance(value, ir.Var) and value.name == previous.target
                        and uses[value.name] == writes[value.name] == 1):
                    break
                statement = replace(statement, **{field: previous.value})
                output.pop()
            output.append(statement)
        return tuple(output)

    return replace(program, body=block(program.body))
