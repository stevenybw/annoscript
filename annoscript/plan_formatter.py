"""Read-only Infer dependency projection and inspection formats.

The projection includes implicit region completion and iteration activation;
Loop.enter must not be confused with Loop.done. Edges are a union of possible
paths, not a new executable schedule. Guards and the selected mode's prerequisites
govern actual readiness; success always awaits checks. Carried edges retain
witnesses rather than inventing a single iteration distance for a transitive
path with multiple recurrences.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass, fields, replace
import html
import json
from pathlib import Path
import re

from annoscript import ir
from annoscript import plan as p
from annoscript.formatter import _expression, _location


@dataclass(frozen=True)
class _Arc:
    source: tuple[str, str]
    target: tuple[str, str]
    kind: str = ""
    carried: str | None = None
    distance: int = 0
    guard: p.Gate | None = None
    hazard: str | None = None
    conservative: bool = False


_NOTES = (
    "节点表示静态 Infer 调用位置；循环可产生多个实例，共享请求可能只发起一次。",
    "依赖图是所有可能路径的并集；调用须满足各自的数据、控制和内存依赖，不执行未选分支。",
    "eager 允许独立任务越过尚未完成的约束检查；strict 保留约束顺序屏障。两者成功返回前都须通过当前路径上的全部检查。",
    "依赖层用于阅读，不是固定调度波次。并行候选是保守子集，不是对实际并发量的预测。",
    "同算子只构成批处理候选：还需同时就绪、模型/系统提示词/后端选项兼容。不同参数可能要求拆批。",
    "严格模式保留单条 request 的失败顺序；跨 request 的调度与批处理不在此图中。",
)


def _prompt_label(expr):
    """Human-readable template, with dynamic parts visibly left unresolved."""
    if isinstance(expr, ir.Literal) and isinstance(expr.value, str):
        return expr.value
    if isinstance(expr, ir.FormatString):
        return "".join(_prompt_label(part) for part in expr.parts)
    if isinstance(expr, ir.FormatValue):
        return "{" + _expression(expr.value) + "}"
    if isinstance(expr, ir.Binary) and expr.op == "add":
        return _prompt_label(expr.left) + _prompt_label(expr.right)
    return "{" + _expression(expr) + "}"


class _View:
    def __init__(self, plan):
        self.plan = plan
        self.tasks = {t.id: t for t in plan.tasks}
        self.values = {v.id: v for v in plan.values}
        self.regions = {r.id: r for r in plan.regions}
        self.requests = {q.id: q for q in plan.requests}
        self.sites = [t for t in plan.tasks if isinstance(t, p.Infer)]
        self.alias = {t.id: f"I{i:02d}" for i, t in enumerate(self.sites, 1)}
        self.loops = {t.id: f"L{i:02d}" for i, t in enumerate((t for t in plan.tasks if isinstance(t, p.Loop)), 1)}
        self.branches = {t.id: f"B{i:02d}" for i, t in enumerate((t for t in plan.tasks if isinstance(t, p.Branch)), 1)}
        self.checks = {t.id: f"C{i:02d}" for i, t in enumerate((t for t in plan.tasks if isinstance(t, p.Check)), 1)}
        self.request_alias = {q.id: f"Q{i:02d}" for i, q in enumerate(plan.requests, 1)}

    def path(self, task):
        return p.loop_path(self.tasks[task].region, self.regions)

    def expand(self, bound):
        """Bounded symbolic substitution for display only; never evaluate code."""
        budget = [160]

        def value(v, seen):
            t = self.tasks[self.values[v].producer]
            if isinstance(t, p.Infer):
                return ir.Var(self.alias[t.id])
            if isinstance(t, p.Eval) and v not in seen and budget[0] > 0:
                return expr(t.expression.expression, {b.name: b.value for b in t.expression.bindings}, seen | {v})
            return ir.Var(self.values[v].name or v)

        def expr(e, bindings, seen):
            budget[0] -= 1
            if isinstance(e, ir.Var):
                return value(bindings[e.name], seen)
            if budget[0] <= 0:
                return e
            def walk(x):
                if isinstance(x, ir.Expr):
                    return expr(x, bindings, seen)
                if isinstance(x, tuple):
                    return tuple(walk(v) for v in x)
                return x
            return replace(e, **{f.name: walk(getattr(e, f.name)) for f in fields(e)})

        return expr(bound.expression, {b.name: b.value for b in bound.bindings}, set())

    def display(self, bound):
        return _expression(self.expand(bound))

    def choices(self, region):
        choices = {}
        while region is not None:
            r = self.regions[region]
            if r.kind in ("then", "else"):
                choices[r.owner] = r.kind == "then"
            region = r.parent
        return choices

    def compatible(self, a, b):
        left, right = self.tasks[a], self.tasks[b]
        if left.request == right.request:
            return False
        x, y = self.choices(left.region), self.choices(right.region)
        return not any(k in y and v != y[k] for k, v in x.items())

    def arcs(self, mode):
        execution_mode = "eager" if mode == "speculative" else mode
        arcs = []

        def add(a, b, kind="", **kwargs):
            arcs.append(_Arc(a, b, kind, **kwargs))

        for d in self.plan.dependencies_for(mode):
            add((d.source, d.port), (d.target, "start"), d.kind,
                carried=d.mapping.carried_loop, distance=d.mapping.distance,
                guard=d.condition, hazard=d.hazard, conservative=d.certainty == "conservative")
        for t in self.plan.tasks:
            add((t.region, "active"), (t.id, "start"), "control")
            if isinstance(t, p.Loop):
                add((t.id, "start"), (t.id, "enter"))
                add((t.id, "enter"), (t.id, "done"), "completion")  # zero iterations
                add((t.id, "enter"), (t.body, "active"), "control")
                for child in self.regions[t.body].tasks:
                    add((child, "done"), (t.id, "done"), "completion")
                    if execution_mode in t.serial_modes:
                        add((child, "done"), (t.body, "active"), "iteration_order", carried=t.id, distance=1)
            elif isinstance(t, p.Join):
                add((t.id, "start"), (t.id, "done"))
                branch = self.tasks[t.branch]
                for region, port in ((branch.then_region, "true"), (branch.else_region, "false")):
                    for child in self.regions[region].tasks:
                        add((child, "done"), (t.id, "done"), "completion", guard=p.Gate(branch.id, port))
                add((t.id, "done"), (t.id, "continue"), "control")
            else:
                add((t.id, "start"), (t.id, "done"))
                if isinstance(t, p.Branch):
                    for region, port in ((t.then_region, "true"), (t.else_region, "false")):
                        add((t.id, "done"), (t.id, port), "control", guard=p.Gate(t.id, port))
                        add((t.id, port), (region, "active"), "control", guard=p.Gate(t.id, port))
        return arcs

    def project(self, mode):
        incoming = defaultdict(list)
        for arc in self.arcs(mode):
            incoming[arc.target].append(arc)
        result = []
        for target in self.sites:
            # Recurrences outside the endpoints' common loop coordinates are
            # internal work (gather/broadcast), not Infer-to-Infer recurrences.
            groups = defaultdict(set)
            for source in self.sites:
                common = []
                for a, b in zip(self.path(source.id), self.path(target.id)):
                    if a != b:
                        break
                    common.append(a)
                groups[tuple(common)].add(source.id)
            for common, sources in groups.items():
                result.extend(self._trace(target, common, sources, incoming))
        return sorted(result, key=lambda e: (e["source"], e["target"], e["carried"]))

    def _trace(self, target, common, sources, incoming):
        # Two states per signal suffice for each shared coordinate prefix.
        # Distances remain on witness arcs, not on a fabricated static DAG.
        result = []
        start = ((target.id, "start"), False)
        seen, queue = {start}, deque([start])
        parents, first, terminals = defaultdict(list), {}, []
        while queue:
            state = queue.popleft()
            signal, carried = state
            if signal[1] == "done" and signal[0] in self.alias:
                if signal[0] in sources:
                    terminals.append(state)
                continue
            for arc in incoming[signal]:
                previous = (arc.source, carried or arc.carried in common)
                parents[previous].append((state, arc))
                if previous not in seen:
                    seen.add(previous)
                    first[previous] = (state, arc)
                    queue.append(previous)
        for terminal in terminals:
            used, todo, visited = set(), [terminal], set()
            while todo:
                state = todo.pop()
                if state == start or state in visited:
                    continue
                visited.add(state)
                for parent, arc in parents[state]:
                    used.add(arc)
                    todo.append(parent)
            witness, state = [], terminal
            while state != start:
                parent, arc = first[state]
                witness.append({"source": list(arc.source), "target": list(arc.target), "kind": arc.kind,
                                "loop": self.loops.get(arc.carried), "distance": arc.distance})
                state = parent
            guards = sorted({f"{self.branches.get(a.guard.task, a.guard.task)}={a.guard.port}" for a in used if a.guard})
            recurrence = sorted({(self.loops[a.carried], a.distance) for a in used if a.carried in common})
            result.append({
                "source": self.alias[terminal[0][0]], "target": self.alias[target.id],
                "carried": terminal[1], "kinds": sorted({a.kind for a in used if a.kind}),
                "hazards": sorted({a.hazard for a in used if a.hazard}),
                "guards": guards, "recurrences": [{"loop": loop, "distance": distance} for loop, distance in recurrence],
                "via_loops": sorted({self.loops[a.carried] for a in used if a.carried and a.carried not in common}),
                "gather": [self.loops[x] for x in self.path(terminal[0][0])[len(common):]],
                "broadcast": [self.loops[x] for x in self.path(target.id)[len(common):]],
                "conservative": any(a.conservative for a in used), "witness": witness,
            })
        return result

    def nodes(self):
        result = []
        for site in self.sites:
            args_task = self.tasks[self.values[site.arguments].producer]
            expression = self.expand(args_task.expression) if isinstance(args_task, p.Eval) else ir.Var(site.arguments)
            arguments = _expression(expression)
            prompt, label = arguments, site.id
            if isinstance(expression, ir.TupleExpr) and len(expression.items) == 2:
                positional, keywords = expression.items
                if isinstance(positional, ir.TupleExpr) and positional.items:
                    instruction = positional.items[0]
                elif "instructions" in site.keywords and isinstance(keywords, ir.TupleExpr):
                    instruction = keywords.items[site.keywords.index("instructions")]
                else:
                    instruction = expression
                prompt = instruction.value if isinstance(instruction, ir.Literal) and isinstance(instruction.value, str) else _expression(instruction)
                label = _prompt_label(instruction)
            request = self.requests[site.request]
            gates = []
            for g in site.gates:
                task = self.tasks[g.task]
                if isinstance(task, p.Branch):
                    gates.append(f"{self.branches[task.id]}={g.port}: {self.display(task.condition)}")
                elif isinstance(task, p.Loop):
                    gates.append(f"{self.loops[task.id]}: iteration active")
                else:
                    gates.append(f"{g.task}.{g.port}")
            result.append({
                "id": self.alias[site.id], "task": site.id, "origin": site.origin,
                "operation": request.operation, "label": " ".join(label.split())[:78],
                "prompt": prompt, "arguments": arguments, "keywords": list(site.keywords),
                "request": self.request_alias[site.request], "reusable": request.reusable,
                "reuse_scope": [self.loops[x] for x in request.scope],
                "loops": [self.loops[x] for x in self.path(site.id)], "gates": gates,
                "source": _location(site.source.span), "calls": [_location(s) for s in site.source.calls],
            })
        return result

    def analyze(self, mode):
        execution_mode = "eager" if mode == "speculative" else mode
        edges = self.project(mode)
        nodes = self.nodes()
        ids = [n["id"] for n in nodes]
        original = {self.alias[s.id]: s.id for s in self.sites}
        adjacency = {n: set() for n in ids}
        for e in edges:
            if not e["carried"]:
                adjacency[e["source"]].add(e["target"])
        reach = {}
        for n in ids:
            seen, todo = set(), list(adjacency[n])
            while todo:
                x = todo.pop()
                if x not in seen:
                    seen.add(x)
                    todo.extend(adjacency[x])
            reach[n] = seen
        for e in edges:
            e["redundant"] = not e["carried"] and any(e["target"] in reach[v] for v in adjacency[e["source"]] if v != e["target"])
        remaining, layers = set(ids), []
        while remaining:
            ready = sorted(n for n in remaining if not any(n in adjacency[k] for k in remaining))
            if not ready:
                break
            layers.append(ready)
            remaining.difference_update(ready)
        rank = {n: i for i, layer in enumerate(layers) for n in layer}
        for node in nodes:
            node["layer"] = rank.get(node["id"], len(layers))

        def groups(members):
            # Deterministic disjoint cliques, deliberately not exponential
            # maximum-antichain enumeration. Exclusive sites never share one.
            buckets = []
            for n in members:
                for bucket in buckets:
                    if all(self.compatible(original[n], original[x]) for x in bucket):
                        bucket.append(n)
                        break
                else:
                    buckets.append([n])
            return [b for b in buckets if len(b) > 1]

        parallel, batches = [], []
        operations = {n["id"]: n["operation"] for n in nodes}
        for i, layer in enumerate(layers):
            parallel.extend({"layer": i, "sites": g} for g in groups(layer))
            for op in sorted(set(operations[n] for n in layer)):
                batches.extend({"layer": i, "operation": op, "sites": g}
                               for g in groups([n for n in layer if operations[n] == op]))
        loops = []
        for tid, alias in self.loops.items():
            loop = self.tasks[tid]
            members = [n["id"] for n in nodes if alias in n["loops"]]
            carried = [e for e in edges if e["carried"] and e["source"] in members and e["target"] in members
                       and any(r["loop"] == alias for r in e["recurrences"])]
            shared = [n["id"] for n in nodes if n["id"] in members and alias not in n["reuse_scope"] and n["reusable"]]
            loops.append({"id": alias, "task": tid, "iterable": self.display(loop.iterable), "sites": members,
                          "serial": execution_mode in loop.serial_modes, "carried_infer_edges": len(carried),
                          "parallel_iterations": bool(members) and not carried and execution_mode not in loop.serial_modes,
                          "shared_sites": shared, "batch_sites": [n for n in members if n not in shared] if not carried and execution_mode not in loop.serial_modes else [],
                          "carried_values": [self.values[t.outputs[0]].name or t.outputs[0] for t in self.plan.tasks
                                             if isinstance(t, p.Carry) and t.loop == tid]})
        checks = [{"id": alias, "task": tid, "condition": self.display(self.tasks[tid].condition),
                   "message": self.display(self.tasks[tid].message) if self.tasks[tid].message else None}
                  for tid, alias in self.checks.items()]
        notes = list(_NOTES)
        speculation = {}
        if mode == "speculative":
            notes[1] = "依赖图保留正式执行的数据、控制和内存依赖；每条输入先完成本轮 Jev 决策，再合批已确认路径的 Qwen 任务。"
            notes.append("Jev 可跨未决分支推测；Qwen 须确认路径。候选仍受参数、内存及循环窗口限制，交替依赖按轮执行。")
            speculation["speculative_candidates"] = [n["id"] for n in nodes if n["reusable"]]
        return {"entry": self.plan.entry, "mode": mode, "nodes": nodes, "edges": edges, "layers": layers,
                "parallel": parallel, "batches": batches, "loops": loops, "checks": checks,
                "unresolved_cycles": sorted(remaining), "notes": notes, **speculation,
                "stats": {"tasks": len(self.plan.tasks), "infer_sites": len(nodes), "requests": len(self.plan.requests),
                          "dependencies": len(self.plan.dependencies_for(mode)), "infer_edges": len(edges)}}


def analyze_plan(plan: p.ExecutionPlan, *, mode: p.FailureMode = "eager") -> dict:
    """Return detached, JSON-compatible inspection data, not a scheduling IR."""
    if not isinstance(plan, p.ExecutionPlan):
        raise TypeError("analyze_plan expects an ExecutionPlan")
    plan.validate()
    plan.dependencies_for(mode)
    return _View(plan).analyze(mode)


def _edge_label(edge):
    label = "/".join(edge["kinds"])
    if edge["hazards"]:
        label += " " + "/".join(edge["hazards"])
    if edge["carried"]:
        label += " carried " + ",".join(f"{r['loop']}:+{r['distance']}" for r in edge["recurrences"])
    if edge["conservative"]:
        label += " (may)"
    return label


def format_plan(plan: p.ExecutionPlan, *, mode: p.FailureMode = "eager", show_source: bool = False) -> str:
    data = analyze_plan(plan, mode=mode)
    lines = [f"Execution Plan v{plan.version} · {plan.entry!r} · mode={mode}",
             f"{len(data['nodes'])} Infer sites / {len(plan.requests)} request definitions / {len(plan.tasks)} tasks", "", "Infer 调用："]
    for n in data["nodes"]:
        lines.append(f"  {n['id']} {n['operation']} [{n['request']}] {n['label']}")
        if n["loops"]:
            lines.append("    loops: " + ", ".join(n["loops"]))
        for gate in n["gates"]:
            lines.append("    gate: " + gate)
        if show_source:
            lines.append(f"    [{n['task']}] {n['source']}")
            lines.extend("    inlined from " + call for call in n["calls"])
    if mode == "speculative":
        lines.extend(("", "两阶段推测候选（先 Jev 跨未决分支决策，再 Qwen 按已确认路径合批）：",
                      "  " + ", ".join(data["speculative_candidates"])))
    lines.extend(("", "Infer 依赖（隐藏同轮传递边；完整路径可在 HTML 中查看）："))
    for e in data["edges"]:
        if not e["redundant"] or e["carried"]:
            lines.append(f"  {e['source']} -> {e['target']} [{_edge_label(e)}]")
    if not data["edges"]:
        lines.append("  无 Infer 间依赖。仍需满足各自的激活条件和非 Infer 前置任务。")
    lines.extend(("", "并行候选（按依赖层；不是执行波次）："))
    lines.extend(f"  层 {g['layer']}: {', '.join(g['sites'])}" for g in data["parallel"])
    if not data["parallel"]:
        lines.append("  未发现同轮静态候选组。")
    lines.extend(("", "批处理候选（需要后端兼容且同时就绪）："))
    lines.extend(f"  {g['operation']} · 层 {g['layer']}: {', '.join(g['sites'])}" for g in data["batches"])
    if not data["batches"]:
        lines.append("  未发现同轮静态候选组。")
    lines.extend(("", "循环："))
    for loop in data["loops"]:
        if not loop["sites"]:
            continue
        status = "可跨迭代并行" if loop["parallel_iterations"] else "存在迭代顺序限制"
        lines.append(f"  {loop['id']} {', '.join(loop['sites'])}: {status}; {loop['iterable']}")
        if loop["batch_sites"]:
            lines.append("    跨迭代批处理候选: " + ", ".join(loop["batch_sites"]))
        if loop["shared_sites"]:
            lines.append("    跨迭代共享请求（不是多次调用待批处理）: " + ", ".join(loop["shared_sites"]))
        if loop["carried_values"]:
            lines.append("    携带值: " + ", ".join(loop["carried_values"]))
    lines.extend(("", "约束检查（strict 顺序屏障）：" if mode == "strict" else "约束检查（成功返回前必须通过）："))
    lines.extend(f"  {c['id']}: {c['condition']}" for c in data["checks"])
    if data["unresolved_cycles"]:
        lines.append("未解析的同轮环（不据此推断并行）: " + ", ".join(data["unresolved_cycles"]))
    lines.extend(("", *data["notes"], "HTML: python3 -m annoscript.inspect PLAN.json --format html -o plan.html"))
    return "\n".join(lines) + "\n"


def format_plan_mermaid(plan: p.ExecutionPlan, *, mode: p.FailureMode = "eager", show_source: bool = False) -> str:
    data = analyze_plan(plan, mode=mode)
    def quote(text):
        # Mermaid quoted labels use decimal entities for delimiters. No source
        # text is allowed to inject a Mermaid statement, URL or HTML element.
        return "".join(c if c.isalnum() or c in " ,.:_-/+" else f"#{ord(c)};" for c in text)
    lines = ["flowchart LR", f"  %% Infer dependency projection, mode={mode}; conditional paths are a union."]
    for n in data["nodes"]:
        text = f"{n['id']} {n['operation']} {n['label']}"
        if n["loops"]:
            text += " · " + ",".join(n["loops"])
        if show_source:
            text += " · " + n["source"]
        lines.append(f'  {n["id"]}["{quote(text)}"]')
    for e in data["edges"]:
        if not e["redundant"] or e["carried"]:
            arrow = "-.->" if e["carried"] else "-->"
            lines.append(f'  {e["source"]} {arrow}|"{quote(_edge_label(e))}"| {e["target"]}')
    return "\n".join(lines) + "\n"


def format_plan_html(plan: p.ExecutionPlan, *, mode: p.FailureMode = "eager", show_source: bool = False) -> str:
    plan.validate()
    plan.dependencies_for(mode)
    view = _View(plan)
    data = {"eager": view.analyze("eager"), "strict": view.analyze("strict"),
            "speculative": view.analyze("speculative"), "initial_mode": mode,
            "show_source": show_source}
    # JSON in an inert script still needs escaping: a prompt can contain </script>.
    payload = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("&", "\\u0026").replace("<", "\\u003c").replace(">", "\\u003e")
    template = Path(__file__).with_name("_plan_viewer.html").read_text(encoding="utf-8")
    replacements = {"__PLAN_TITLE__": html.escape(plan.entry, quote=True), "__PLAN_DATA__": payload}
    return re.sub(r"__PLAN_(?:TITLE|DATA)__", lambda match: replacements[match[0]], template)
