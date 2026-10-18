"""Hard structure/count/nonempty constraints; advisory code-point lengths, with source-isolated Substring rules."""

import json

from .base import InferenceError


# Keep a complete JSON code point in one lexical FSM: a repeated independent
# json-char rule makes every text token context-sensitive in XGrammar.
JSON_ATOM = r'(?:[^"\\\x00-\x1f]|\\["\\/bfnrt]|\\u(?:[0-9a-cA-Ce-fE-F][0-9a-fA-F]{3}|[dD][0-7][0-9a-fA-F]{2})|\\u[dD][89aAbB][0-9a-fA-F]{2}\\u[dD][c-fC-F][0-9a-fA-F]{2})'


def json_body(nonempty):
    return "Regex(" + json.dumps(JSON_ATOM + ("+" if nonempty else "*")) + ")"


def literal(value):
    return json.dumps(value, ensure_ascii=False)


def repeat(rule, minimum, maximum):
    if maximum == 0:
        return '""'
    return rule + ('{' + str(minimum) + ',' + ('' if maximum is None else str(maximum)) + '}')


def ensure_satisfiable(call):
    a = call.arguments
    if call.operation == "extract" and not a["source"] and a.get("min_items", 0) > 0:
        raise InferenceError("empty extraction source cannot satisfy min_items",
                             code="unsatisfiable_constraints", call_id=call.id,
                             metadata={"dispatch_status": "not_sent"})


def grammar(calls, *, batch=False, extract_mode="validated"):
    definitions = []
    cache = {}

    def value_rule(call):
        ensure_satisfiable(call)
        a = call.arguments
        key = (call.operation, a.get("source"), a.get("min_items", 0), a.get("max_items"),
               bool(a.get("min_length", 0)), bool(a.get("min_length_per_item", 0)))
        if key in cache:
            return cache[key]
        prefix = 'value-' + str(len(cache))
        cache[key] = prefix
        if call.operation == "generate":
            body = json_body(bool(a.get('min_length', 0)))
            definitions.append(prefix + ' ::= "\\\"" ' + body + ' "\\\""')
            return prefix
        if call.operation == "extract" and not a["source"]:
            definitions.append(prefix + ' ::= "[]"')
            return prefix
        item = prefix + '-string'
        if call.operation == "extract" and extract_mode == "substring_chars":
            chunks = [literal(json.dumps(c, ensure_ascii=False)[1:-1]) for c in a['source']]
            options = ', allow_empty=false'
            body = prefix + '-body'
            definitions.append(body + ' ::= Substring(' + ', '.join(chunks) + options + ')')
        else:
            minimum = 1 if call.operation == "extract" else a.get('min_length_per_item', 0)
            body = json_body(bool(minimum))
        definitions.append(item + ' ::= "\\\"" ' + body + ' "\\\""')
        low, high = a.get('min_items', 0), a.get('max_items')
        if high == 0:
            interior = '""'
        else:
            tail = repeat('(\",\" ' + item + ')', max(0, low - 1), None if high is None else high - 1)
            interior = item + ' ' + tail
            if low == 0:
                interior = '(' + interior + ')?'
        definitions.append(prefix + ' ::= "[" ' + interior + ' "]"')
        return prefix

    if batch:
        parts = [literal('{"answers":{')]
        for i, call in enumerate(calls):
            field = 'text' if call.operation == 'generate' else 'items'
            prefix = (',' if i else '') + json.dumps(call.id, ensure_ascii=False)
            prefix += ':{"type":' + json.dumps(call.operation) + ',"' + field + '":'
            parts.extend([literal(prefix), value_rule(call), literal('}')])
        parts.append(literal('}}'))
        root = ' '.join(parts)
    else:
        call, = calls
        ensure_satisfiable(call)
        if call.operation == 'generate':
            # Standalone generate remains raw text, with no JSON wrapper.
            root = 'Regex(' + json.dumps(r'[\s\S]' + ('+' if call.arguments.get('min_length', 0) else '*')) + ')' 
        else:
            root = literal('{"items":') + ' ' + value_rule(call) + ' ' + literal('}')
    return 'root ::= ' + root + '\n' + '\n'.join(definitions)
