"""Client-side example snapshots and compilation; no backend dependencies."""

from copy import deepcopy
import json
import unittest
from unittest.mock import patch


from annoscript_examples.fixtures import REVIEW_INPUT as REVIEW, LAUNCH_INPUT as LAUNCH


from annoscript.compiler import compile_program
from annoscript.ir import Program
from annoscript.anno import Example, ExtractResponse, GenerateResponse, GenerateListResponse
from annoscript._test_ir_runtime import evaluate


LOCATION = Example(LAUNCH, ["广州"])


def captured(anno):
    return anno.extract("地点", source_field="text", max_items=2,
                       max_length_per_item=80, examples=[LOCATION]).items


def constructed(anno):
    state = {"text": "示例在广州发布"}
    answer = ["广州"]
    example = Example(state=state, answer=answer)
    state["text"] = "示例条目"
    answer[0] = "条目"
    return anno.extract("地点", source_field="text", max_items=2,
                       max_length_per_item=80, examples=[example]).items


def different_examples(anno):
    # Deliberately vary only the answer to check identity, not business accuracy.
    first = anno.choice("情感", {"正面": "积极", "中性": "中立"},
                       examples=[Example(REVIEW, "正面")]).choice
    second = anno.choice("情感", {"正面": "积极", "中性": "中立"},
                        examples=[Example(REVIEW, "中性")]).choice
    third = anno.choice("情感", {"正面": "积极", "中性": "中立"},
                       examples=[Example(REVIEW, "正面")]).choice
    return [first, second, third]


def named_generations(anno):
    title = anno.generate("标题", max_length=60, field_name="event_title").text
    dimensions = anno.generate_list("维度", max_items=8, max_length_per_item=24,
                                   field_name="tags").items
    locations = anno.extract("地点", source_field="text", max_items=8,
                            max_length_per_item=64, field_name="locations").items
    return [title, dimensions, locations]


def nullable_example(anno):
    value = anno.generate("缺依据为 null", nullable=True, min_length=1, max_length=24,
                         examples=[Example(state="示例正文", answer=None)]).text
    if value is not None:
        return value.strip()
    return value


def requirements_for_all_operations(anno):
    requirements = ["仅依据正文。", "保留原意。"]
    a = anno.noul("判断", requirements=requirements).noul
    b = anno.choice("选择", {"yes": "是"}, requirements=requirements).choice
    c = anno.score("评分", ["低", "高"], requirements=requirements).score
    d = anno.generate("标签", max_length=24, requirements=requirements).text
    e = anno.generate_list("维度", max_items=2, max_length_per_item=24,
                          requirements=requirements).items
    f = anno.extract("地点", source_field="text", max_items=2, max_length_per_item=24,
                    requirements=requirements).items
    return [a, b, c, d, e, f]


class ExampleClientTests(unittest.TestCase):
    def test_requirements_survive_client_ir_for_all_operations(self):
        from annoscript.anno import NoulResponse, ChoiceResponse, ScoreResponse

        class Probe:
            def __init__(self):
                self.calls = []

            def record(self, operation, options, response):
                self.calls.append(operation)
                assert options["requirements"] == ["仅依据正文。", "保留原意。"]
                return response

            def noul(self, instructions, **options):
                return self.record("noul", options, NoulResponse(.8))

            def choice(self, instructions, criteria, **options):
                return self.record("choice", options, ChoiceResponse("yes", .9, {"yes": 1.}))

            def score(self, instructions, criteria, **options):
                return self.record("score", options, ScoreResponse({0: 0., 1: 1.}, 1., dict(enumerate(criteria)), .9))

            def generate(self, instructions, **options):
                return self.record("generate", options, GenerateResponse("标签"))

            def generate_list(self, instructions, **options):
                return self.record("generate_list", options, GenerateListResponse(["维度"]))

            def extract(self, instructions, **options):
                return self.record("extract", options, ExtractResponse(["广州"]))

        for optimize in (False, True):
            probe = Probe()
            program = Program.from_json(compile_program(requirements_for_all_operations, optimize=optimize).to_json())
            self.assertEqual(evaluate(program.to_dict(), probe), [.8, "yes", 1., "标签", ["维度"], ["广州"]])
            self.assertEqual(probe.calls, ["noul", "choice", "score", "generate", "generate_list", "extract"])

    def test_nullable_example_and_none_survive_client_ir(self):
        class Probe:
            state = "正文"

            def generate(self, instructions, **options):
                assert options["nullable"] is True
                example = options["examples"][0]
                example = example.to_dict() if type(example) is Example else example
                assert example["answer"] is None
                return GenerateResponse(None)

        for optimize in (False, True):
            program = Program.from_json(compile_program(nullable_example, optimize=optimize).to_json())
            self.assertIsNone(evaluate(program.to_dict(), Probe()))

    def test_field_names_survive_client_ir_roundtrip(self):
        class Probe:
            state = {"text": "广州"}

            def generate(self, instructions, **options):
                return GenerateResponse(options["field_name"])

            def generate_list(self, instructions, **options):
                return GenerateListResponse([options["field_name"]])

            def extract(self, instructions, **options):
                return ExtractResponse([options["field_name"]])

        for optimize in (False, True):
            program = Program.from_json(compile_program(named_generations, optimize=optimize).to_json())
            self.assertEqual(evaluate(program.to_dict(), Probe()),
                             ["event_title", ["tags"], ["locations"]])


if __name__ == "__main__":
    unittest.main()
