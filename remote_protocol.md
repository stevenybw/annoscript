# Lev IR HTTP 协议 v1

本文定义 `lev.remote.RemoteLevClient` 与未来执行服务的接口。本仓库当前提供客户端和
离线测试服务替身，没有可部署的正式 HTTP 执行服务。测试替身只绑定本机回环地址，
仅用于 Python 3.10 → Python 3.12 的跨进程验收。

## 执行边界

```text
Python 3.10+ 客户端：输入快照及 precondition → 编译 → IR 优化 → HTTP
Python 3.12+ 执行端：协议及 IR 校验 → 服务端优化 → lowering → Runtime → 模型后端
```

IR 是跨端契约；Execution Plan、任务 ID、依赖图和调度实现不进入 HTTP 协议。
客户端默认沿用 `compile_program(..., optimize=True)`，`optimize=False` 仅关闭客户端优化，
不禁止服务端再次优化。服务端优化必须保持 IR 的程序、约束及执行模式语义。
`mode` 是 `eager` 或 `strict`，不是对服务端内部 Plan 结构的要求。

Python `precondition` 不会进入 IR。远程客户端先检查整批输入，再发送合法记录；
单条输入错误保留原始索引并在本地返回 `invalid_input`。检查器异常或编译失败阻止整批 HTTP 提交。
直接使用 HTTP 协议的调用方必须自行执行等价的应用输入检查。未来服务端还必须独立校验
请求、IR 版本与结构，并实施执行资源限制；不能把 `Program.from_json()` 的结构检查视为
完整的安全性、类型或资源证明。本轮不传输 Python 校验函数，也不实现生产级服务入口。

## 请求

`POST {base_url}/v1/runs:batch`，`Content-Type: application/json`，
`Accept: application/x-ndjson`。`base_url` 可以包含路径前缀，不能带凭据、查询参数或片段。
提供 `auth_token` 时发送 `Authorization: Bearer ...`；该 token 用于执行服务，模型凭据留在服务端。
客户端不读取 `.credential`，也不跟随 HTTP 重定向。

请求为以下五个字段组成的 JSON 对象：

| 字段 | 含义 |
| --- | --- |
| `protocol_version` | 整数 `1`，独立于 IR 的版本号 |
| `ir` | `Program.to_dict()`，`format="lev-ir"`、`version=1` |
| `ir_sha256` | 完整 IR 的 SHA-256 十六进制摘要 |
| `mode` | `eager` 或 `strict` |
| `inputs` | 对象数组，每项只有 `index`、`state`；索引保留原输入位置，允许不连续 |

完整请求示意如下：

```json
{
  "protocol_version": 1,
  "ir": {"format": "lev-ir", "version": 1, "...": "完整的 Program 字段"},
  "ir_sha256": "完整 IR 的摘要",
  "mode": "eager",
  "inputs": [{"index": 2, "state": {"text": "示例"}}]
}
```

示意中的 `ir` 不是可执行样例，实际请求包含完整的现有 IR。`state` 必须为字符串、数组或对象，
内部仅允许有限 JSON 值且对象键必须是字符串。每个 `index` 必须是唯一非负整数。

摘要计算与 `lev.remote.ir_sha256()` 一致：对 IR 使用
`json.dumps(ir, sort_keys=True, ensure_ascii=False, allow_nan=False, separators=(",", ":"))`，
再编码为 UTF-8 并计算 SHA-256。保留数组顺序，包括 IR 中表示字典条目的有序数组。
摘要包含源码定位，因此不同 Python 版本的等价 IR 可能具有不同摘要。
它表示提交内容，不表示语义等价，也不包含模型、Runtime 或发布版本。

每批内联 IR，没有自动注册、缓存或重试。空批次和全部无效的输入不发送 HTTP。

## 响应

成功建立流使用 HTTP `200` 和 `Content-Type: application/x-ndjson`。每行一个 UTF-8 JSON 对象，
以 LF 或 CRLF 结束，不使用空白行心跳；拒绝重复 JSON 键、非有限数字和未终止的行。
v1 帧字段固定；拓展字段需要更新协议。服务端必须在 `done` 后结束响应。

```json
{"type":"accepted","protocol_version":1,"ir_sha256":"请求中的摘要"}
{"type":"result","index":2,"status":"success","value":null,"error":null}
{"type":"done"}
```

- `accepted` 必须在任何结果之前出现且只出现一次，客户端核对协议版本和 IR 摘要。
- `result` 必须包含 `type/index/status/value/error`。成功的 `error` 必须为 `null`，`value` 可为任意有限 JSON 值，包含 `null`。
- 单条失败使用 `status="error"`、`value=null`、`error={"code":"非空错误代码","message":"说明"}`；同批其他记录仍可成功。
- 每个已提交索引恰好返回一次；重复、未提交索引或布尔索引均为协议错误。
- `done` 只有 `type` 字段，要求所有已提交索引完成，随后必须正常结束响应。缺少 `done`、缺少结果或结束后追加帧均报错。
- 提交级失败可以在任何时刻返回 `{"type":"error","error":{"code":"backend_unavailable","message":"说明"}}`，
  包括尚未发送 `accepted` 时；客户端立即抛出异常，不再期待 `done`。

HTTP 非 `200` 响应表示提交被拒绝，可返回 `{"error":{"code":"错误代码","message":"说明"}}`。
非结构化错误体不会作为诊断直接暴露，客户端保留 HTTP 状态和 `Retry-After` 原始值。

## 客户端错误与资源所有权

| 异常 | 行为 |
| --- | --- |
| `RemoteSubmissionError` | HTTP 拒绝或 `error` 帧；提供 `code`、`http_status`、`retry_after` |
| `RemoteTransportError` | 连接失败、网络中断、HTTP 阶段超时；服务端可能已经开始执行 |
| `RemoteProtocolError` | 非法或不完整的响应、版本/摘要不符、索引错误 |

客户端不自动重试，也不将缺失记录补成成功空值。已经交付的结果不会撤回；调用方仍需完整消费
流才能确认批次正常结束，提前退出不代表其余记录成功。`ordered=True` 仅改变交付顺序，
可能缓存后完成的记录；默认按结果可用顺序交付，本地输入错误优先交付。

`timeout=300.0` 是 HTTP 各阶段超时，不是整个任务的总执行时限。关闭 stream 会关闭响应，
关闭 client 会关闭其所有 stream 和连接池；均可重复关闭。提前关闭不保证远程推理已取消。
客户端生命周期和单个 stream 的迭代由一个线程管理；不同工作线程使用独立客户端。

正式服务、持久化幂等、发布版本绑定、生产资源预算和 Spark 适配另行实现；协议摘要不能替代发布版本。
