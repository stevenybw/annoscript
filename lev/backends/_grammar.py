"""Compact JSON structure for list operations, independent of business contracts.

Only JSON syntax is constrained here. Runtime validates the decoded strings,
item counts, character lengths and verbatim extraction, including empty values.
The grammar is source, not generated from the car example or a third-party repo.
"""

ITEMS_GRAMMAR = r'''
root ::= "{\"items\":[" (string ("," string)*)? "]}"
string ::= "\"" character* "\""
character ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
'''.strip()


BATCH_GRAMMAR = r'''
root ::= "{\"answers\":{" (answer ("," answer)*)? "}}"
answer ::= string ":" (text-answer | list-answer)
text-answer ::= "{\"type\":\"generate\",\"text\":" string "}"
list-answer ::= "{\"type\":" ("\"generate_list\"" | "\"extract\"") ",\"items\":[" (string ("," string)*)? "]}"
string ::= "\"" character* "\""
character ::= [^"\\\x00-\x1f] | "\\" (["\\/bfnrt] | "u" hex hex hex hex)
hex ::= [0-9a-fA-F]
'''.strip()
