"""Shared structural analyses; none of these helpers evaluate user code.

Pure IR expressions may still raise, read mutable data, or allocate objects.
Safety below concerns supported IR values, excluding resource exhaustion.
"""

from collections import Counter
from dataclasses import fields, replace

from annoscript import ir


def map_value(value, transform):
    if isinstance(value, ir.Expr):
        return transform(value)
    if isinstance(value, tuple):
        return tuple(map_value(item, transform) for item in value)
    return value


def rewrite_expr(expr, transform):
    children = {f.name: map_value(getattr(expr, f.name), lambda e: rewrite_expr(e, transform))
                for f in fields(expr)}
    return transform(replace(expr, **children))


def map_expressions(statement, transform):
    """Map immediate expression fields, never bodies or constraint expressions."""
    if isinstance(statement, ir.Constrain):
        return statement
    return replace(statement, **{
        f.name: map_value(getattr(statement, f.name), transform) for f in fields(statement)
    })


def expression_reads(expr):
    result = Counter()

    def visit(node):
        if isinstance(node, ir.Var):
            result[node.name] += 1
        return node

    rewrite_expr(expr, visit)
    return result


def direct_reads(statement):
    result = Counter()

    def visit(expr):
        result.update(expression_reads(expr))
        return expr

    # Constraints participate in liveness even though transformations skip them.
    for f in fields(statement):
        map_value(getattr(statement, f.name), visit)
    return result


def target_names(target):
    if isinstance(target, ir.NameTarget):
        return {target.name}
    if isinstance(target, ir.TupleTarget):
        return set().union(*(target_names(item) for item in target.items))
    raise TypeError(f"Unknown target: {type(target).__name__}")


def definitions(statement):
    if isinstance(statement, (ir.Assign, ir.Infer)):
        return {statement.target}
    if isinstance(statement, (ir.Unpack, ir.For)):
        return target_names(statement.target)
    return set()


def child_blocks(statement):
    if isinstance(statement, ir.If):
        return (statement.then_body, statement.else_body)
    if isinstance(statement, ir.For):
        return (statement.body,)
    return ()


def walk_statements(body):
    for statement in body:
        yield statement
        for block in child_blocks(statement):
            yield from walk_statements(block)


def block_writes(body):
    return set().union(*(definitions(s) for s in walk_statements(body)))


def literal_key(value):
    # Python equality conflates bool/int/float and the signs of floating zero.
    return (type(value), value.hex() if type(value) is float else value)


def atom_key(expr):
    if isinstance(expr, ir.Literal):
        return (ir.Literal, literal_key(expr.value))
    if isinstance(expr, ir.Var):
        return (ir.Var, expr.name)
    raise TypeError("Expected a literal or variable")


def resolve(expr, facts):
    seen = set()
    while isinstance(expr, ir.Var) and expr.name in facts and expr.name not in seen:
        seen.add(expr.name)
        expr = facts[expr.name]
    return expr


def invalidate(facts, names):
    """Forget copies of old bindings as well as the overwritten bindings."""
    resolved = {name: resolve(value, facts) for name, value in facts.items()}
    return {name: value for name, value in resolved.items()
            if name not in names and not (isinstance(value, ir.Var) and value.name in names)}


def join_facts(left, right):
    return {name: value for name, value in left.items()
            if name in right and atom_key(value) == atom_key(right[name])}


def join_exits(left, left_falls, right, right_falls):
    if not left_falls:
        return right if right_falls else {}
    if not right_falls:
        return left
    return join_facts(left, right)


def bound_before(program):
    """Definitely bound names at each statement, including loop body entry."""
    points = {}

    def block(body, incoming):
        bound = set(incoming)
        falls = True
        for statement in body:
            points[statement.id] = frozenset(bound)
            if isinstance(statement, ir.If):
                a, af = block(statement.then_body, bound)
                b, bf = block(statement.else_body, bound)
                bound = a & b if af and bf else a if af else b if bf else set()
                falls = af or bf
            elif isinstance(statement, ir.For):
                block(statement.body, bound | definitions(statement))
                # The loop can execute zero times.
            elif isinstance(statement, ir.Return):
                falls = False
            else:
                bound |= definitions(statement)
            if not falls:
                break
        return bound, falls

    block(program.body, set())
    return points


def discardable(expr, bound):
    """Prove that evaluating and discarding an expression cannot fail or act."""
    if isinstance(expr, (ir.Literal, ir.State)):
        return True
    if isinstance(expr, ir.Var):
        return expr.name in bound
    if isinstance(expr, (ir.ListExpr, ir.TupleExpr)):
        return all(discardable(item, bound) for item in expr.items)
    if isinstance(expr, ir.DictExpr):
        # Only literal keys are known to be hashable without a type analysis.
        return all(isinstance(key, ir.Literal) and discardable(value, bound)
                   for key, value in expr.entries)
    return False
