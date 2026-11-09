# AnnoScript 持久化任务协议 v1

Python 3.10–3.14 客户端提交 `lev-ir` version 1；Python 3.12–3.14 服务端校验、优化、lowering，
再通过真实 Runtime 执行。所有 HTTP 交互是普通 JSON 短请求；不传 Python 源码、Python 检查函数、
Execution Plan 或模型凭据。保留单一源码包，服务器依赖单独安装。

这是单机 SQLite 首版。覆盖连接丢失和 API/worker 进程重启，不提供机器故障自动接管、
生产部署、安全多租户执行沙箱或 Spark 端到端集成。IR 结构校验不等于完整安全证明。

## 两层客户端

推荐普通调用方使用 `AnnoClient.remote(base_url, auth_token=None, *, mode="eager", optimize=True,
timeout=30.0, transport=None)`，然后调用与本地相同的 `submit(program, inputs, ordered=False)`。
先执行整批输入快照、precondition 和编译，再建立会话和提交合法输入。全部非法或空批次零 HTTP；
单条非法输入作为本地 `invalid_input` 合并，保留原始索引。检查器缺陷或编译失败直接中止提交。
`transport` 主要供离线测试使用。客户端和迭代器由单个用户线程调用；内部心跳线程独立续租。

客户端会话、幂等键、原请求、任务编号、结果页和游标仅存内存。每次 `submit()` 创建一个新任务；
同一次调用的重试复用同一请求和键。每页先完整检查，再缓存并推进游标；异常页不能部分交付。
`ordered=True` 缓存乱序结果，可能需要与整批结果大小相当的内存。结果保持 `Result` 的原值，
成功的 `None`、数组和对象不被转换。消费结果不意味着外部业务输出已可靠落地。

需要显式持久任务管理时使用底层接口：

```python
from annoscript.compiler import compile_program
from annoscript.jobs import AnnoJobClient
from annoscript._client_common import prepare_inputs
from annoscript_examples.request import annotate

valid, local_errors = prepare_inputs(annotate, inputs)
ir = compile_program(annotate)
with AnnoJobClient(base_url, auth_token=service_token) as client:
    # 上层必须处理 local_errors；只有 valid 非空时创建任务。
    job = client.submit_ir(
        ir, [{"index": index, "state": state} for index, state in valid],
        idempotency_key="caller-owned-stable-key",
        mode="eager",
    )
    page = client.fetch_results(job, cursor=None, limit=100)

# 新进程使用相同主体凭据，显式找回任务，不重新编译。
with AnnoJobClient(base_url, auth_token=service_token) as client:
    job = client.lookup("caller-owned-stable-key")
    status = client.status(job)
    # 持久任务暂停后由调用方决定何时恢复。
    if status["state"] == "paused":
        job = client.resume(job, pause_generation=status["pause_generation"],
                            idempotency_key="stable-key-for-this-resume")
```

`submit_ir` 接受 `Program` 或 IR 对象，以及带 `index/state` 的合法输入；可选 `release_id`、
`application_revision` 和内部 `session_id`。底层方法返回 JSON 字典；任务参数可以传状态字典或 `job_id`。
`AnnoJobClient.close()` 只关闭 HTTP 连接，持久任务继续存在。调用方管理幂等键和业务消费 checkpoint，
库不创建本地数据库，不提供消费 ACK。读取游标丢失后可重新读取已有结果，不重新推理。

precondition 不在 IR 中。底层 `submit_ir` 不执行 Python 业务前置检查，直接使用它或 HTTP 的调用方
必须先执行等价输入校验。可选业务版本是调用方元数据，服务端不会据此认定校验已运行。

## HTTP 接口

所有接口使用 `Authorization: Bearer ...`，认证主体隔离任务和幂等键。客户端不跟随重定向。
请求和成功响应使用 `application/json`；拒绝重复 JSON 键、非有限数字和非法结构。
错误响应为 `{"error":{"code":"...","message":"..."}}`，错误不返回模型响应正文或凭据。

| 方法与路径 | 请求 | 响应 |
| --- | --- | --- |
| `POST /v1/jobs` | `Idempotency-Key`；下面的任务请求 | `202`，`{protocol_version, job}` |
| `GET /v1/jobs:lookup` | `Idempotency-Key` | `{protocol_version, job}` |
| `GET /v1/jobs/{id}` | 无 | `{protocol_version, job}` |
| `GET /v1/jobs/{id}/results?cursor=...&limit=100` | 初始 cursor 省略，limit 1–1000 | 结果页 |
| `POST /v1/jobs/{id}:resume` | `Idempotency-Key`；`{pause_generation}` | `{protocol_version, job}` |
| `POST /v1/jobs/{id}:cancel` | 空对象 | `{protocol_version, job}`，取消已持久化 |
| `GET /v1/releases/{id}` | 无 | `{protocol_version, release_id, manifest, available}` |
| `POST /v1/sessions` | `Idempotency-Key`；空对象 | `201`，会话响应 |
| `POST /v1/sessions/{id}:heartbeat` | 空对象 | 会话响应 |
| `POST /v1/sessions/{id}:close` | 空对象 | 会话响应，取消已持久化 |

任务请求的必填字段为 `protocol_version: 1`、`ir`、`ir_sha256`、`mode` 和
`inputs: [{index, state}, ...]`。输入索引必须是唯一非负整数，允许不连续；state 为字符串、数组或对象。
可选字段为 `release_id`、`application_revision`、`session_id`，均为非空字符串。
输入为空不创建任务。默认每个任务最多 500 条输入、请求最多 32 MiB，由执行版本资源策略绑定。

幂等键为 1–256 字符。同一认证主体内，同键同请求返回原任务；同键不同请求 `409`。
IR 摘要沿用 UTF-8、排序对象键、紧凑分隔符、拒绝非有限数值的规范 JSON SHA-256；
包含源码位置，因此不是语义等价标识。请求指纹额外包含输入、模式和可选字段，保留输入对象键顺序，
不把可被 IR 遍历观察的顺序变化视为同一请求。HTTP JSON 空白和协议顶层键顺序不影响指纹。

任务状态 `job` 包含 `job_id/ir_sha256/release_id/mode/input_indices/state/counts`、
`pause_generation/pause_reason`、`created_at/updated_at`。计数为 `total/completed/success/error`；
时间为 Unix 秒。执行版本在首次接受时固定；幂等重放先找已有任务，再考虑当前默认版本。

结果页包含 `protocol_version/job/results/next_cursor/has_more/done`。每项结果为
`{seq,index,status,value,error,attempts}`，其中 `status/value/error` 遵循现有 Result 契约。
`seq` 是任务内结果提交顺序，从 1 开始；`index` 是原始输入索引。
游标绑定任务与已读取序号，非法、其他任务或超过日志末尾的游标被拒绝。
结果与状态来自同一事务快照。空页可以表示还没有新结果；只有任务终结且当前页读到末尾才 `done=true`。
读取不删除结果，重复读取同一游标不会重新执行任务。

## 状态、重试与取消

- `queued/running`：待执行或正在执行；每条输入由独立 attempt 运行，默认并发 8。
- `paused`：保留所有已完成结果，等待显式恢复，不自动清理。临时模型错误每轮最多 3 次 attempt，
  耗尽后暂停整个任务；认证、配置或已确认的聊天模板协议错误立即暂停。暂停后不启动新 attempt，
  已领取的 attempt 也不得发送新模型请求；在途请求仍可保存响应和计量，不撤销其所有权。
- `completed`：全部输入有终态结果，允许其中包含逐条错误。违反输出契约直接返回错误，不修改输出或重复修复。
- `cancelled/failed`：取消或提交级失败。已完成结果保留，其余输入得到明确错误结果，迟到执行不能覆盖终态。

默认重试模型连接失败、超时和 HTTP `408/429/5xx`；保留结构化状态与 `Retry-After`。
HTTP 400 中已确认的模板布局错误由 HTTP 后端识别为 `backend_protocol_error`；
其余 400（包括上下文超限）为单条失败，不暂停任务。
release policy 的 `retry_unknown_outcomes` 默认为 `true`，保留通用重试行为；显式设为 `false`
时，无明确 HTTP 响应的模型超时、断线及失联 worker 的过期 attempt 暂停任务，不自动重提未知结果。
有明确 HTTP 响应的可重试错误仍受原次数上限约束。
指数退避加入随机抖动并尊重 `Retry-After`；默认模型阶段超时 60 秒，每条 attempt 执行上限 300 秒。
执行上限耗尽作为 `execution_timeout` 终态错误，区别于可重试的模型网络超时。
worker 崩溃后状态不确定的 attempt 也消耗预算。`resume` 使用暂停代次和独立幂等键，
已有 attempt 尚未退出时返回 `409 attempts_active`；恢复只重置未完成输入的本轮预算，不清零累计次数。
已终态的逐条错误不重新执行。原执行版本不可用时暂停，不能换版本恢复。

简化客户端对网络中断、阶段超时及 `408/429/5xx` 最多尝试 5 次、总重连预算 120 秒；
默认 HTTP 阶段超时 30 秒，普通轮询等待不消耗重连预算，也不表示任务总超时。
认证和协议错误不重试。`RemoteSubmissionError` 保留 code、HTTP 状态和可用的 `Retry-After`；
网络耗尽为 `RemoteTransportError`；非法响应、摘要/版本/索引/游标错误为 `RemoteProtocolError`。
执行暂停抛出 `RemoteJobPausedError`，随后关闭并取消该提交；它不自动重置服务端重试预算。

`results.close()` 请求取消该任务；`client.close()` 和退出上下文关闭整个所有者会话。
提交结果无法确认时也关闭整个会话，从而覆盖客户端还不知道 job_id 的已接受任务；
无法确认单个任务取消时停止拥有整个会话，避免无限续租孤儿任务，因此同一客户端其他未完成任务也会取消。
显式关闭最多等待 5 秒确认持久化取消，失败抛出 `RemoteCancellationError`；清理异常不覆盖已有业务异常。
`weakref.finalize` 析构时最多尝试 1 秒；后台线程和结果迭代器不保活客户端外壳。

会话每 30 秒续租，默认租期 300 秒。API/worker 均处理过期会话，持久化关闭并取消其未完成任务；
关闭、过期后禁止迟到提交和续租，重复创建使用原幂等键不会复活会话。
进程被强制终止时不能依赖析构，租约是兜底机制。取消阻止后续调度和结果提交，已经发送的模型调用
只能尽力中止。单机服务不可用期间无法立即执行回收，恢复时先处理过期会话。

## 存储与执行版本

SQLite 使用 WAL、外键及 `synchronous=FULL`。数据库事务不等待模型；接受任务与输入、幂等键、
版本绑定原子持久化后返回 `202`。worker 租约默认 60 秒、每 15 秒续租，结果写入同时检查 fencing token、
租约有效期、任务及会话状态。模型请求与 SQLite 提交没有分布式事务：崩溃可能重复模型调用，
但每个输入只有一个不可变终态结果。当前新建数据库 schema 为 2；历史 v1 必须显式离线迁移，不自动清空或升级数据库。

执行版本 ID 是不可变清单的 SHA-256，绑定源码 build 摘要、Python 次版本、关键依赖版本、
模型地址/名称/部署修订、优化开关、Runtime 配置及资源/重试策略。每次 API/worker 启动从相同配置
构建可用版本清单；代码或配置改变后，旧任务不会悄悄在新版本执行。`releases` 命令列出实际 ID。
恢复旧任务需要重新提供匹配版本的执行环境，首版不自动下载、启动或调度历史发布。
模型修订由部署方保证指向相应不可变部署；版本绑定不承诺外部模型输出具有确定性。
模型凭据不进入清单或 SQLite，允许独立轮换。

`generation.profile` 省略时规范化为 `legacy`，可显式选择 `qwen_vllm`；`decision` 不接受该字段。
profile 与 `disable_thinking` 都进入清单，紧凑 grammar 的源码由 `build_sha256` 绑定。
相同源码下省略 profile 与显式 `legacy` 产生相同版本；切换 profile 或 thinking 开关产生新版本。
worker 只使用任务绑定的清单，不受本地 `ANNO_GENERATE_PROFILE` 或后续默认 release 变化影响。
旧服务配置仍能加载；升级源码会改变执行版本，不会自动迁移旧任务或清空数据库。

`decision.request_adapter` 和 `generation.request_adapter` 分别选择请求构造方案。
保持原行为的默认方案为 `decision_native_v1` / `generation_chat_v1`；省略时也把对应默认值
写入 manifest，同一源码下显式默认与省略产生相同 release。未知方案或跨阶段方案拒绝加载。
适配器实现由 `build_sha256` 绑定，worker 只使用清单中的选择，不受本机
`ANNO_DECISION_REQUEST_ADAPTER` / `ANNO_GENERATE_REQUEST_ADAPTER` 影响。
模型、协议 profile、生成策略和原始 confidence 含义均不因该配置新增而改变。
远程任务/IR 无需新增字段。离线请求预览见 [README](README.md#离线预览实际请求)。

Decision 可显式选择 `decision_configurable_v1`，并用 `decision.request_adapter_options`
配置 default/tasks。任务按 field_name 精确匹配；各方案完整声明 criteria/examples/state，
格式见 [README](README.md#按任务选择-decision-表示)。规范化配置进入 manifest 和 release 摘要，
配置/适配器源码改变均产生新 release；旧任务保留原绑定。
worker 仅使用 manifest，不读取 `ANNO_DECISION_REQUEST_ADAPTER_OPTIONS` 覆盖。
原样默认不新增 options 字段；生成侧不接受该字段。模型及校准配置不随表示方案自动改变。

终态任务完整数据保留 7 天；维护时删除输入、IR、attempt 和结果正文，保留幂等墓碑。
之后查询或同键重放返回 `410 job_expired`，不会误创建新任务。未终结的持久任务不因 TTL 被清理。

## 服务配置与本地启动

在源码目录安装 `requirements-server.txt`，API 与 worker 都使用 Python 3.12+。
下面是配置结构示例，地址、模型名称和修订必须由部署方填写，不会自动启动模型服务：

```json
{
  "auth": {"local-user": "ANNO_SERVICE_TOKEN"},
  "default_release": "main",
  "releases": {
    "main": {
      "decision": {
        "base_url": "https://decision.example/v1/decisions",
        "model": "decision-model",
        "revision": "immutable-decision-deployment-1"
      },
      "generation": {
        "base_url": "https://generation.example/v1",
        "model": "generation-model",
        "revision": "immutable-generation-deployment-1"
      },
      "server_optimize": true,
      "disable_thinking": true
    }
  }
}
```

`auth` 将稳定主体名称映射到服务 token 的环境变量名；任务归属于主体，token 可轮换。
API 与 worker 的配置加载要求这些服务 token 已设置。worker 另外读取
`ANNO_DECISION_AUTH_TOKEN`、`ANNO_GENERATE_AUTH_TOKEN`，不自动读取 `.credential` 或执行 shell 文件。

```sh
python3.12 -m pip install -r requirements-server.txt
python3.12 -m annoscript.server releases --config /absolute/path/service.json
# 两个独立终端，共享同一本地持久数据库和相同源码/配置
python3.12 -m annoscript.server api --db /absolute/path/lev-jobs.sqlite --config /absolute/path/service.json
python3.12 -m annoscript.server worker --db /absolute/path/lev-jobs.sqlite --config /absolute/path/service.json
```

API 默认 `127.0.0.1:8000`；可显式设置 `--host/--port`，worker 可设置 `--concurrency`。
资源策略可在 release 的 `policy` 中覆盖默认值，Runtime 选项可在 `runtime` 中覆盖；
`policy.retry_unknown_outcomes` 只接受布尔值，参与 release 摘要。
LLM 合批默认配置为 `max_llm_batch_calls=8`、`max_llm_batch_input_bytes=262144`、
`max_llm_batch_output_tokens=16384`、`llm_batch_default_tokens=1024`、`llm_batch_repair_attempts=1`。
这些选项进入 release manifest 和摘要，改动会生成新 release，已有任务保留原绑定。
设 `max_llm_batch_calls=1` 关闭合批；补救次数仅允许 0 或 1，按 attempt 内每个逻辑调用计数，
独立于服务端 attempt 重试。详见[预算、Backend 与失败语义](README.md#llm-有界合批与补救)。
每个 attempt 的 `max_active_runs` 固定为 1，由 worker 控制输入间并发。
这些配置变化产生新的 release ID。请勿把 SQLite 放在网络文件系统上作为多机调度数据库。

Qwen/vLLM 的 release 配置示例：在上述 `main` 中将 `generation` 改为以下对象，
保留 `disable_thinking: true`，地址、名称与不可变部署修订仍由部署方填写：

```json
{
  "base_url": "https://qwen-vllm.example/v1",
  "model": "qwen-model",
  "revision": "immutable-qwen-deployment-1",
  "profile": "qwen_vllm"
}
```

该 profile 使用 `chat_template_kwargs.enable_thinking=false` 和列表/抽取的紧凑
`structured_outputs.grammar`；`disable_thinking: false` 则省略 thinking 覆盖，使用服务默认行为。
`generate` 保持纯文本，现有 token 预算不变。服务拒绝 grammar 时明确报错，不自动降级；
模板布局错误暂停任务。配置只保存在服务端，IR、任务 HTTP 请求与简化客户端接口均不增加字段。
请求格式和独立 grammar 验收命令见[项目 README](README.md#qwenvllm-请求适配)。
离线替身测试不证明某个实际模型、chat template 或 vLLM 部署配置已经兼容。

## 离线验收

```sh
# Python 3.10–3.14；只需 requirements.txt，不导入执行后端
ANNO_RUN_LIVE_TESTS=0 python -m annoscript.client_tests -q
# Python 3.12–3.14；需要 requirements-server.txt
ANNO_RUN_LIVE_TESTS=0 python -m unittest discover -q
# 两个解释器缺一即失败，不静默跳过
python3.12 -m annoscript.job_cross_version_tests \
  --client-python /path/to/python3.10 --server-python /path/to/python3.12
git diff --check
```

跨进程验收使用独立 API/worker、临时 SQLite 和确定性模型替身，覆盖两种执行模式、
客户端优化开关、有序/乱序结果、丢失提交和结果响应、API/worker 重启及客户端被强制结束。
合成示例中的抽取跨度必须来自当前输入原文。

## 外部管线绑定与版本查询

`AnnoJobClient.release(release_id)` 对现有 `GET /v1/releases/{id}` 提供 Python 3.10+ 封装，
返回 `protocol_version/release_id/manifest/available`，检查 ID 与 manifest 的规范化摘要。
它不导入 Runtime、Backend、Plan、lowering 或 Server。返回 `available=false` 的历史清单
可以解释旧结果，不表示可以用该版本继续执行。HTTP 协议没有新增字段。

需要 driver 重启恢复的批量管线使用底层持久任务，不创建简化客户端会话。固定 release，
先保存请求指纹、稳定幂等键和输入映射，再提交 IR；找回任务时保留原版本与索引。
已有接受记录却找不到任务、幂等墓碑 `410`、结果或版本绑定变化都应停止，不换键重发。
输入检查继续由客户端 `prepare_inputs()` 执行，不上传 Python precondition。

跨批缓存可以在客户端内存保存成功结果，同时在外部 checkpoint 保存原任务／结果索引
及摘要。结果本身仍由 Server 的持久任务保存；缓存引用恢复不延长终态任务默认 7 天的
保留期。保存绑定后遇到过期引用需显式处理，不能悄悄切换为新推理。既有业务输出 outbox
与 AnnoScript 执行 Journal 的职责不同，前者仍可保存准备写入业务 sinks 的最终快照。

任务 `paused` 由操作方修复原因后，携带当前 `pause_generation` 和稳定的恢复操作幂等键
显式 `resume()`；管线不会自动恢复预算。底层客户端 `close()` 只释放连接，不取消任务。
此路径不改变 `AnnoClient.remote()` 的自动会话取消语义，也不表示业务系统已部署接入。

## 执行指标与显式数据库迁移

新增 `GET /v1/jobs/{id}/metrics`，Python 接口 `AnnoJobClient.metrics(job)`。沿用认证、归属隔离、
重试及 7 天保留规则，过期返回 410。提交/状态/结果分页字段完全保留；Result/ResultError 和 IR 不变。
该接口返回一致事务快照：

```text
protocol_version: 1
metrics_version: 1
job: 原状态对象（含不可变执行绑定）
provisional: 非终态为 true，暂停也属于暂态
summary: 任务级汇总
inputs: [{index, metrics: 与 summary 同结构的逐输入汇总}]
```

汇总含 history_complete、attempts（数量、状态计数、耗时）、queue_wait_seconds、backends（decision/generation）。
后端汇总含 requests、confirmed_requests、unknown_requests、total_requests、questions、confirmed_questions、
total_questions、http_seconds、input_tokens、output_tokens。Decision 或 LLM 一次批量只计一次请求；
questions 按逻辑成员数计，单发补救为 1；整批 token 只记录一次，不摊分或复制到每个成员。
Runtime 请求复用不产生新 HTTP 计量。

数值统计包含 observed（已知小计）、total（完整总计，否则 null）、known/unknown；耗时另有 p50/p95，
用量另有 sources/reasons。所有失败及重试 attempt 累计统计，显式恢复不清零。
input_tokens/prompt_tokens、output_tokens/completion_tokens 是别名；冲突、非整数、负值、缺失不估算。
嵌套 reasoning/cache 与 total token 不重复相加。并发 HTTP 时间之和不是墙钟时间，也不代表 GPU 计算时间。
首次排队耗时仅在同一系统 boot 标识下用持久化单调时间计算；历史或跨 boot 无法比较时为 null。

请求发送前记录意图，响应后保存受限用量和 HTTP 状态/单调耗时。不存在跨数据库与模型的原子事务：
未确认意图可能尚未发送，也可能模型已经计算；因此不能把请求意图数量当成完整计费数量。
每次写入验证 attempt fencing、租约和任务/会话有效性；重复结束不重复计数，失效 worker 不补写。
已经观察到的 HTTP 结果尽量在 Runtime 取消该调用前完成落盘，所有权被取消后仍拒绝迟到写入。
可选的 `Worker(request_observer=...)` 接收按 request ID 关联的意图与结束事件，包含阶段、attempt、
时间、状态和服务实际返回的 usage；结束事件另含 `storage_committed`。即使持久化期间取消，
已观察到的响应也会通知观察器。观察器异常以 `metrics_failed` 暂停任务，不触发额外模型重试。
该接口默认关闭，不改变 HTTP API 或数据库格式；观察器应自行幂等落盘且不得记录认证信息。

指标不保存提示词、输入正文、模型响应正文、认证信息或完整 trace。标准 HTTP Backend 自动采集；
自定义非 HTTP 后端的历史完整性为 false，不伪装成零费用。数据缺失与任务成功是独立概念。
迁移前的任务 history_complete=false，不能从旧结果反推出模型消耗。

需要本地故障诊断时，可显式提供 `Worker(diagnostic_observer=...)`；默认关闭，独立于数字计量。
HTTP Backend 也接受同名可选参数。观察器收到 request/response/outcome 事件及每个 attempt 的
runtime trace/plan；原始正文使用 bytes，request ID 与计量一致，包含逻辑调用和补救父请求 ID。
响应在解析和业务校验前观察，认证头及密钥不传给观察器。调用方负责仅在受控本地目录保存正文，
不得将这些诊断事件混入常规指标 API 或业务输出。诊断写入失败以 `diagnostics_failed` 暂停，
不额外重试模型；已收到响应的状态与用量仍先完成记账。取消或断线未得到正文时不虚构响应。

### 离线迁移 v1 → v2

先完成或显式处理在途任务，并停止所有 API/worker。确认独立备份路径尚不存在，再执行：

```sh
python3.12 -m annoscript.server.migrate --db /absolute/path/lev-jobs.sqlite \
  --backup /absolute/path/lev-jobs-v1.backup.sqlite
```

工具先通过 SQLite backup（包含已提交 WAL）备份，检查完整性并同步，再事务性增加指标字段/表。
失败回滚，已有备份不覆盖；未知版本和非 v1 库拒绝。服务启动仍不自动迁移，旧 v1 实现也不能打开 v2 库。
迁移保留已有任务、结果、幂等键和 release，不重新推理、不改变版本、不回填零指标。
本轮测试只使用临时数据库，不执行任何现存数据库迁移。

代码升级产生新的 release。迁移不能让旧 release 在新实现中执行；已完成结果仍可读取，
未完成旧任务保持原绑定，原执行环境不可用时暂停。工具不自动安装历史 worker 或更换任务版本。

离线跨进程验证命令见 [README](README.md#offline-verification)。API 和 worker 的启动方式保持不变。
