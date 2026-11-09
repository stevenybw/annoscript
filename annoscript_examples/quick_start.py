"""Run the packaged synthetic document using the official anonymous gateway."""
import json
import sys
from annoscript.client import AnnoClient
from annoscript_examples.request import annotate

SAMPLE = {'text': 'The community library opens a quiet reading room on Saturday.', 'kind': 'note'}


def main():
    try:
        with AnnoClient.server() as client:
            with client.submit(annotate, [SAMPLE]) as results:
                for result in results:
                    if result.status != 'success':
                        print('AnnoScript: ' + result.error.message, file=sys.stderr)
                        return 1
                    print(json.dumps(result.value, ensure_ascii=False, indent=2))
        return 0
    except Exception as exc:
        print(str(exc), file=sys.stderr)
        return 1


if __name__ == '__main__':
    raise SystemExit(main())
