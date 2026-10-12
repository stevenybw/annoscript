"""Remove proven dead paths and redundant boolean completion guards."""

from dataclasses import replace

from lev import ir
from ._analysis import block_writes, definitions, invalidate, join_exits, resolve


def _completion_pair(first, second, facts):
    if not (isinstance(first, ir.If) and not first.else_body and first.then_body
            and isinstance(second, ir.If) and not second.else_body):
        return False
    guard = second.condition
    if not (isinstance(guard, ir.Unary) and guard.op == "not" and isinstance(guard.value, ir.Var)):
        return False
    name = guard.value.name
    initial = resolve(ir.Var(name), facts)
    last = first.then_body[-1]
    return (isinstance(initial, ir.Literal) and initial.value is False
            and isinstance(last, ir.Assign) and last.target == name
            and isinstance(last.value, ir.Literal) and last.value.value is True)


def simplify_control_flow(program: ir.Program) -> ir.Program:
    def block(body, incoming):
        facts = dict(incoming)
        output = []
        index = 0
        falls = True
        while index < len(body):
            statement = body[index]
            if index + 1 < len(body) and _completion_pair(statement, body[index + 1], facts):
                statement = replace(statement, else_body=body[index + 1].then_body)
                index += 1
            index += 1
            if isinstance(statement, ir.If):
                condition = resolve(statement.condition, facts)
                if isinstance(condition, ir.Literal):
                    selected = statement.then_body if condition.value else statement.else_body
                    nested, facts, falls = block(selected, facts)
                    output.extend(nested)
                else:
                    yes, a, af = block(statement.then_body, facts)
                    no, b, bf = block(statement.else_body, facts)
                    output.append(replace(statement, then_body=yes, else_body=no))
                    facts = join_exits(a, af, b, bf)
                    falls = af or bf
            elif isinstance(statement, ir.For):
                facts = invalidate(facts, definitions(statement) | block_writes(statement.body))
                nested, _, _ = block(statement.body, facts)
                output.append(replace(statement, body=nested))
            else:
                output.append(statement)
                if isinstance(statement, ir.Assign):
                    value = resolve(statement.value, facts)
                    facts = invalidate(facts, {statement.target})
                    if isinstance(value, ir.Literal):
                        facts[statement.target] = value
                elif isinstance(statement, ir.Constrain):
                    facts.clear()
                else:
                    facts = invalidate(facts, definitions(statement))
                falls = not isinstance(statement, ir.Return)
            if not falls:
                break
        return tuple(output), facts, falls

    return replace(program, body=block(program.body, {})[0])
