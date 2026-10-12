"""Structured backward liveness, preserving evaluations that may fail or act."""

from dataclasses import replace

from lev import ir
from ._analysis import bound_before, definitions, direct_reads, discardable


def eliminate_dead_assignments(program: ir.Program) -> ir.Program:
    points = bound_before(program)

    def block(body, outgoing):
        live = set(outgoing)
        output = []
        for statement in reversed(body):
            reads = set(direct_reads(statement))
            if isinstance(statement, ir.Assign):
                if statement.target not in live and discardable(statement.value, points.get(statement.id, ())):
                    continue
                live = (live - {statement.target}) | reads
            elif isinstance(statement, ir.If):
                yes, a = block(statement.then_body, live)
                no, b = block(statement.else_body, live)
                statement = replace(statement, then_body=yes, else_body=no)
                live = a | b | reads
            elif isinstance(statement, ir.For):
                # A back edge reaches another target binding or exits the loop.
                head = set(live)
                while True:
                    nested, before_body = block(statement.body, head)
                    updated = live | (before_body - definitions(statement))
                    if updated == head:
                        break
                    head = updated
                statement = replace(statement, body=nested)
                live = head | reads
            elif isinstance(statement, ir.Return):
                live = reads
            else:
                live = (live - definitions(statement)) | reads
            output.append(statement)
        return tuple(reversed(output)), live

    return replace(program, body=block(program.body, set())[0])
