"""Compact JSON structure for list operations, independent of business contracts.

JSON syntax and batch ID/type mapping are constrained here. Runtime validates
decoded strings, item counts, character lengths and verbatim extraction.
The grammar is source, not generated from the car example or a third-party repo.
"""

import json


# Keep the string continuation in the same right-recursive rule. Repeating a
# separate `character` rule is language-equivalent but makes xgrammar 0.2.3
# classify tokens against the surrounding parser stack at every decode step
# with the Qwen3.5 vocabulary. It can dominate decoding and cause HTTP timeouts.
# This spelling preserves JSON escapes, Unicode and empty strings; it neither
# narrows extraction semantics nor changes any business/token budget.
STRING_RULES = r'''
string ::= "\"" string-body
string-body ::= "\"" | [^"\\\x00-\x1f] string-body | "\\" escape string-body
escape ::= ["\\/bfnrt] | "u" hex hex hex hex
hex ::= [0-9a-fA-F]
'''.strip()

ITEMS_GRAMMAR = r'''root ::= "{\"items\":[" (string ("," string)*)? "]}"''' + "\n" + STRING_RULES


BATCH_GRAMMAR = r'''
root ::= "{\"answers\":{" (answer ("," answer)*)? "}}"
answer ::= string ":" (text-answer | list-answer)
text-answer ::= "{\"type\":\"generate\",\"text\":" string "}"
list-answer ::= "{\"type\":" ("\"generate_list\"" | "\"extract\"") ",\"items\":[" (string ("," string)*)? "]}"
'''.strip() + "\n" + STRING_RULES


def _extract_rules(source, prefix):
    """Atomic JSON encodings preserve code-point boundaries after JSON decoding.

    Requires the pinned XGrammar allow_empty extension. No Unicode normalization
    or union of sources is permitted. Runtime still checks counts and lengths.
    """
    items, body = prefix + "-items", prefix + "-body"
    if not source:
        return items, [items + ' ::= "[]"']
    chunks = [json.dumps(json.dumps(char, ensure_ascii=False)[1:-1], ensure_ascii=False)
              for char in source]
    string = prefix + "-string"
    return items, [
        body + " ::= Substring(" + ", ".join(chunks) + ", allow_empty=false)",
        string + ' ::= "\\\"" ' + body + ' "\\\""',
        items + ' ::= "[" (' + string + ' ("," ' + string + ')*)? "]"',
    ]


def extract_grammar(source):
    items, rules = _extract_rules(source, "extract")
    return 'root ::= "{\\\"items\\\":" ' + items + ' "}"\n' + "\n".join(rules)


def batch_grammar(calls, *, extract_mode="validated"):
    """Bind every requested ID to its operation, once, in request order.

    A generic union allows an extract ID to produce generate/text, even though
    that answer is valid JSON. The response parser still validates arbitrary
    object ordering and all constraints when the provider ignores the grammar.
    """
    parts = [json.dumps('{"answers":{')]
    sources, rules = {}, []
    for index, call in enumerate(calls):
        field = "text" if call.operation == "generate" else "items"
        prefix = ("," if index else "") + json.dumps(call.id, ensure_ascii=False)
        prefix += ':{"type":' + json.dumps(call.operation) + ',"' + field + '":'
        value_rule = "string" if field == "text" else "items"
        if call.operation == "extract" and extract_mode == "substring_chars":
            source = call.extraction_source
            if source not in sources:
                rule, definitions = _extract_rules(source, f"source-{len(sources)}")
                sources[source] = rule
                rules.extend(definitions)
            value_rule = sources[source]
        parts.extend((json.dumps(prefix, ensure_ascii=False), value_rule, json.dumps("}")))
    parts.append(json.dumps("}}"))
    grammar = "root ::= " + " ".join(parts) + '\nitems ::= "[" (string ("," string)*)? "]"\n' + STRING_RULES
    return grammar + ("\n" + "\n".join(rules) if rules else "")
