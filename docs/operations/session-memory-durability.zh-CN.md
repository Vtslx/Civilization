# Session 记忆持久化与抗灾

**状态：** 已实现并验证
**适用范围：** 所有启用 `persist_sessions` 的服务实例（默认开启）

## 问题

此前 session 记忆只存在于服务进程内存中。服务重启——发布、崩溃、被 OOM 杀掉——所有会话都会丢失。global store 虽然配置了文件路径，但实际没有写入，因此跨会话状态同样会在重启后消失。

## 设计

读路径仍然走内存，写路径变持久：

```
请求 ──► 内存 store ──►（读路径不变：检索、排序、注入）
              │
              └─ 变更 ──► journal 追加（os.write，按会话追加写）
                              │
                   ┌──────────┴──────────┐
                   │  按 fsync 策略提交   │  interval | every_write | never
                   └──────────┬──────────┘
                              │
              每 N 条记录压缩 ──► snapshot.json + journal 重置
                              │
              启动/首次访问 ──► 水合 = snapshot + journal 尾部
```

四条性质让恢复过程明显正确：

1. **追加立刻离开进程。** 记录用 `O_APPEND` 描述符上的 `os.write` 写入。因此**进程**被异常杀死不会丢掉任何已确认的写入——字节已经在操作系统的页缓存里。
2. **提交策略界定丢失窗口。** 后台线程按 `fsync_interval_seconds`（默认 0.25 秒）调用 `fsync`，也可以选择每次写入都提交或从不提交。**断电**最多丢失未提交的尾部。
3. **每条记录都是幂等 upsert。** 在快照之上重放 journal 不会重复或损坏状态，因此"快照已写、journal 尚未截断"之间崩溃是无害的。
4. **残缺的末行会被忽略。** 崩溃产生的半行会被识别并丢弃，它之前的完整记录全部保留。

### 什么情况下丢什么

| 事件 | 影响 | 原因 |
|---|---|---|
| 正常关闭 | 不丢 | `close()` 会提交并 flush |
| 服务重启（发布、升级） | 不丢 | journal 在磁盘上，启动时重放 |
| 进程被杀（`kill -9`、OOM） | 已确认的写入不丢 | 记录已通过 `os.write` 离开进程 |
| 机器断电 / 硬重启 | 最多丢失未提交尾部 | 由 fsync 策略界定（默认 ≤ 0.25 秒的写入） |
| 磁盘损坏 | 本地状态丢失 | 没有副本；需要可靠性请把 `state_dir` 放在有冗余的卷上 |

### 可选的持久化 sidecar 进程

`civilization persist-sidecar` 把 journal 与提交循环放进**独立进程**，服务通过 UNIX socket 发送记录。

它带来的：fsync 工作移出服务进程；持久流在服务重启期间保持开启，因此重启不会打断提交周期。

它**不**带来的：额外的数据安全。数据安全来自 journal 加 fsync 策略，而不是"由哪个进程写"。两个进程在同一台机器上，断电行为一致。

如果 sidecar 不可达（没启动、崩溃、socket 路径过长），服务**不会让记忆写入失败**：它会降级为同一 state 目录下的本地进程内 journal，并在状态里报告 `degraded: true`、原因以及降级起始时间。

sidecar 恢复后客户端会**自动重新挂接**：下一次写入会尝试连接，后台探测线程每 `sidecar_reconnect_interval_seconds`（默认 5 秒）重试一次。交接是安全的——同一时刻只有一个写入者持有 journal 文件：本地 sink 会先提交并关闭，socket 才接管；两者追加的是同一批文件，因此降级期间写入的内容不会丢失。状态里的 `reconnects` 记录交接次数。

## 配置

`EmbeddedConfig` 字段（以及对应的 `civilization serve` 参数）：

| 字段 | 参数 | 默认 | 含义 |
|---|---|---|---|
| `persist_sessions` | `--persist-sessions / --no-persist-sessions` | `True` | 是否在 `state_dir` 下持久化 session 记忆 |
| `persistence_mode` | `--persistence-mode {in_process,sidecar}` | `in_process` | 由谁持有文件 |
| `fsync_policy` | `--fsync {every_write,interval,never}` | `interval` | 提交策略 |
| `fsync_interval_seconds` | `--fsync-interval` | `0.25` | 提交间隔 |
| `snapshot_every_records` | `--snapshot-every` | `512` | journal 压缩阈值 |
| `sidecar_socket` | `--sidecar-socket` | 无 | sidecar 端点，sidecar 模式必填 |
| `sidecar_reconnect_interval_seconds` | — | `5.0` | 降级后重试 sidecar 的间隔 |
| `persist_traces` | `--persist-traces` | `False` | 是否同时持久化记忆 trace 事件 |

state 目录下的布局：

```
<state_dir>/
  sessions/
    <session_id>/
      journal.ndjson     追加写记录
      snapshot.json      压缩后的状态
  var/                 任务、结果、导出（未变）
```

## 运维操作

```bash
# 查看：磁盘上的会话、待提交记录数、提交水位、模式、降级标志
curl -s localhost:8765/admin/orion/memory | jq '.persistence'

# 强制提交（例如在对卷做快照前）
curl -s -X POST localhost:8765/admin/orion/memory/flush | jq '.persistence.sink'

# 使用 sidecar 运行
civilization persist-sidecar --state-dir var/state --socket /tmp/civilization.sock &
civilization serve --persistence-mode sidecar --sidecar-socket /tmp/civilization.sock \
  --provider-base-url <url> --provider-model <model>

# 最强持久性、写入代价最高
civilization serve --fsync every_write --provider-base-url <url> --provider-model <model>
```

请监控 `persistence.sink.pending_records`：它是已确认但尚未提交的记录数。该值持续增长说明磁盘跟不上写入速度。

## 已验证的行为

自动化覆盖（`tests/engine/test_session_persistence.py`）：

- 重启前写入的会话在重启后可读，链接与单元更新一并恢复，id 计数器继续递增而不冲突；
- journal 的残缺末行被忽略，其前的记录全部保留；
- 在快照之上重放陈旧 journal 记录，既不会重复单元也不会重复链接；
- `every_write` 与 `interval` 两种策略最终都能达到 `pending_records == 0`；
- sidecar 能接收记录，在服务被异常杀死后排空并落盘，之后可从文件恢复状态；
- sidecar 不可达时降级为本地写入，而不是让写入失败；
- 恶意 session id 无法逃出 state 目录；
- 未调用显式 save 的 global（跨会话）记忆在重启后仍存在；global store 的 load 会替换内容，且重放不会复活被移除的内容；
- 请求作用域内写入的记录在作用域退出前已进入 journal，嵌套作用域共享同一缓冲，且缓冲中的记录同样触发压缩；
- sidecar 消失后客户端会自动重新挂接（无需重启服务），并保留降级期间写入的记录；
- 保留策略会清理过期会话、始终保留最近 N 个、在达到阈值时压缩 journal 且不丢单元、遵守 `dry_run`，并且从不触碰受保护的会话——通过在线端点同样如此。

端到端崩溃测试（服务使用 `fsync=interval`，间隔 0.1 秒）：

1. 服务向一个会话写入 3 条记忆；
2. 用 `kill -9` 杀掉进程——没有正常关闭，没有调用 flush；
3. 新的服务进程使用同一个 state 目录启动；
4. `read_memory` 返回全部 3 条记忆，`cell_id` 完全一致。

## 写入代价

一次决策请求会写入多个单元与链接（工作、情景、程序性记忆，它们之间的链接，以及可能触发的 replay 巩固）。这些记录会被合并：服务在请求内缓冲，请求结束时对每个会话只发一次 append。持久性不变——记录在响应发出前已经落盘，请求中途崩溃只会丢失尚未确认的工作。

在本实现上的实测：开启 replay 巩固的 3 次决策产生 21 条记录、仅 4 次 append（每次 append 5.25 条），相比每条记录单独写入**减少了 81% 的 journal 写调用**。`POST /admin/orion/memory` 会按会话报告 `records_written`、`appends` 与 `records_per_append`，因此这个比例在生产中可观测，而不是靠假设。

## 保留与压缩

journal 会自动压缩（`snapshot_every_records`，默认 512），因此 journal 始终很小；快照大小与**存活状态**成正比，而不是与历史成正比。要限制多会话下的磁盘占用，有两种方式：

```bash
# 离线：直接操作文件（先停服务）
civilization prune --state-dir var/state --older-than-days 30 --keep 50 --dry-run
civilization prune --state-dir var/state --older-than-days 30 --compact-over 200

# 在线：服务会保护它当前正在服务的会话
civilization prune --base-url http://127.0.0.1:8765 --older-than-days 30 --keep 50
```

也可以直接调用端点：

```bash
curl -s -X POST localhost:8765/admin/orion/memory/retention \
  -d '{"older_than_days": 30, "keep": 50, "compact_over_records": 200}' | jq
```

| 参数 | 含义 |
|---|---|
| `older_than_days` | 丢弃最后写入早于该天数的会话 |
| `keep` | 无论多老，始终保留最近活跃的 N 个会话 |
| `compact_over_records` | journal 记录数达到该值的会话执行合并 |
| `dry_run` | 只报告将要发生什么，不做任何修改 |

在线路径**不会**清理服务内存中持有的会话（包括 global store），并在决策前先提交待写入记录。离线路径没有这些信息——请在服务停止时运行，或改用在线路径。

## 离线记忆备份与恢复

日志重放可恢复进程和主机重启，但磁盘丢失还需要独立故障域中的第二份副本。
备份命令离线运行：创建归档前需停止服务及持久化 sidecar，避免复制期间日志
集合发生变化。先安装加密可选依赖，并从批准的密钥管理器向每个命令进程注入
`CIVILIZATION_BACKUP_PASSPHRASE`；CLI 不接受命令行密钥参数。
密钥必须至少 16 个 UTF-8 字节且具备足够随机性，不要使用字典短语。

```bash
# 核心包保持仅依赖标准库；备份功能需要安装加密 extra。
python -m pip install 'astreusn-civilization-v1[secure-backup]'

# 在实时状态目录之外创建并验证权限为 0600 的加密归档。
civilization backup --state-dir var/state --output /secure-backups/orion-2026-10-26.enc

# 验证所有认证数据块、清单和载荷校验和。
civilization verify-backup --backup /secure-backups/orion-2026-10-26.enc

# 恢复到新目录；已存在的目标目录会被拒绝。
civilization restore --backup /secure-backups/orion-2026-10-26.enc --destination var/state-restored

# 检查恢复结果后，再以恢复目录启动服务。
civilization serve --state-dir var/state-restored --provider-base-url <url> --provider-model <model>
```

归档包含 `sessions/*/snapshot.json`、`sessions/*/journal.ndjson`，以及存在时的
`global-memory.json`。它不包含任务队列、任务结果、审计日志、导出文件、服务商
配置或模型文件。清单保存相对路径、字节数和 SHA-256 校验和。恢复会拒绝未知
版本、路径穿越、链接、意外文件和校验不匹配，并在原子提升已验证的暂存目录后
完成。目标路径必须不存在。新归档使用带版本号的 AES-256-GCM 封装，每个 1 MiB
数据块独立认证，并通过 Argon2id 从密钥派生加密密钥；每个归档使用新的 salt
和 nonce 前缀。校验过程直接流式读取认证数据块，不会把整份明文 tar 写入临时
文件。恢复时将文件写入权限受限的暂存目录，只有最终认证帧及清单检查全部通过
后才会提升。JSON 报告会标记 `encrypted: true`。

旧版 v1 tar 归档仍可不带密钥验证和恢复，但它们是明文，只校验载荷校验和；新建
归档一律加密。恢复加密归档必须提供原密钥；密钥丢失后无法恢复，因此请将密钥
与备份分开存入批准的密钥管理器。恢复时需为私有暂存目录预留足够的目标状态空间。
迁移旧版 v1 时，先恢复到一个新状态目录，再从该目录创建新的加密备份；原明文
归档应按保留策略妥善保护或退役。

记忆可能含有用户隐私数据。请将归档存到源卷之外，并按运维策略保留多个版本，
定期在临时目录中演练恢复。`civilization replicate-backup` 可将已加密归档复制到
不同文件系统；`civilization replicate-s3-backup` 通过可选 `backup-s3` extra 使用
S3-compatible versioned bucket，持久保存 multipart checkpoint，并对远端 VersionId
执行回读与 AEAD 校验。SDK 凭据来自标准 credential chain，备份密钥仍只从环境变量
读取。`civilization s3-backup-retention` 默认只生成 dry-run 计划，显式 `--apply` 才会
删除符合保留策略且未受 Object Lock/legal hold 保护的旧版本。

这些命令不配置定时任务、告警或异地目标。必须对获准的远端 bucket 完成实时恢复演练，
才能将该路径视为异地灾备；同一主机或同一物理站点的另一个文件系统不能防范站点级故障。
S3 bucket 还应配置超过预期重试/续传窗口的 incomplete multipart abort lifecycle 规则。
适配器可通过 ListMultipartUploads 恢复唯一的 content-addressed 上传；同一 key 存在多个
未完成上传时会失败关闭，需人工清理。

## 边界与非目标

- **没有自动复制。** 日志与快照能在本机重启后恢复。备份命令可生成可迁移副本，但不会自动调度或将它传到其他故障域。
- **global（跨会话）记忆走同一套 journal。** 它在启动时水合，每次变更都会追加，因此不再依赖显式 save。`POST /admin/orion/global-store/save|load` 保留为导出/导入：load 会替换内容，并记录为一条 reset 加若干 upsert，因此重放不会把被 load 移除的内容复活。
- **trace 默认不持久化。** 决策真正依赖的状态是单元与链接；诊断用的 trace 流需显式开启（`persist_traces`）。
- **同一 state 目录只允许一个写入者。** 两个服务共享目录会交错写入 journal。请改用 sidecar 模式或分开目录。
- **归档不压缩。** 日志会合并到快照，旧会话可通过 `civilization prune` 清理；备份 tar 未压缩，命令也不备份任务或导出数据。
