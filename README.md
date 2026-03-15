# fubot

`fubot` 是一个本地优先、面向实际落地的多通道 AI 助手框架。它保留了 CLI 和聊天入口的直接使用体验，同时在内部引入了正式的多代理运行时，由协调器负责任务分派、执行角色选择、模型与 Provider 路由，并将工作流状态持续写入工作区。

当前版本已经完成了本地 Agent 框架最核心的一层能力建设，覆盖通道接入、工具调用、记忆管理、定时任务、Provider 接入和工作流编排，适合作为可扩展、可私有化的本地智能体底座。

## 功能说明

### 1. 多代理运行时

- 使用 `Coordinator + Executors` 架构处理任务，不再只是单个代理循环。
- 默认内置 5 类执行角色：
  - `generalist`：通用任务和用户沟通
  - `builder`：编码、修改、重构
  - `researcher`：搜索、证据收集、信息整理
  - `verifier`：测试、校验、代码审查
  - `operator`：定时任务、运维和执行型操作
- 支持任务分类、执行器选择、模型路由、并行执行和最终结果汇总。

### 2. 工作流与状态持久化

- 保存 workflow、task、assignment、execution log、shared board 和 provider health。
- 保留追加式会话记录、长期记忆、历史归档和上下文整理能力。
- 适合需要持续运行、可恢复、可审计的本地 Agent 场景。

### 3. 工具能力

- 文件系统：读写、编辑、列目录
- Shell：受限执行命令
- Web：搜索与抓取
- Message：主动发消息
- Spawn：后台任务
- Cron：定时任务
- MCP：动态工具接入

这些工具还能按执行角色做权限隔离，不同执行器只拿到自己需要的工具集合。

### 4. 多模型与多 Provider

- 支持 OpenAI 兼容接口直连
- 支持 LiteLLM 路由多 Provider
- 支持 Azure OpenAI
- 支持 OAuth 类 Provider
- 支持按任务动态选择模型，并结合健康缓存做回退

### 5. 多通道接入

当前仓库内已实现或保留的通道包括：

- Telegram
- Discord
- Slack
- WhatsApp
- 飞书
- 钉钉
- QQ
- 企业微信
- Matrix
- Mochat
- Email

### 6. 技能系统

- 支持技能目录、`SKILL.md`、frontmatter 元数据
- 已内置 `github`、`weather`、`summarize`、`tmux`、`clawhub`、`skill-creator`、`memory`、`cron` 等技能
- 可按需扩展本地技能

## 当前已实现的功能

结合当前仓库代码、测试和运行时配置，`fubot` 目前已经实现的核心能力包括：

- CLI 单次执行与交互模式
- Gateway 常驻运行模式
- 多通道消息接入与主动消息发送
- 文件系统、Shell、Web、Cron、Spawn、MCP 等工具能力
- 追加式会话、长期记忆、历史归档与上下文整理
- 多 Provider 接入与模型回退
- 工作流状态持久化与执行日志保存
- 技能系统与本地技能扩展

如果按 OpenClaw 官方当前已经公开实现的功能来对照，`fubot` 目前已经覆盖了下面这些同类能力：

| 功能类别 | OpenClaw | `fubot` |
| --- | --- | --- |
| CLI / Gateway | 有 `gateway`、`agent`、wizard、doctor 等命令面 | 已实现 `onboard`、`agent`、`gateway`、`status`、`channels status`、`provider login` |
| 多聊天通道 | 覆盖 WhatsApp、Telegram、Slack、Discord、Matrix、Feishu 等大量通道 | 已实现 Telegram、Discord、Slack、WhatsApp、飞书、钉钉、QQ、企业微信、Matrix、Mochat、Email |
| 技能系统 | 已有 skills、bundled/workspace skills | 已实现技能目录、`SKILL.md`、frontmatter 元数据与内置技能 |
| 定时与自动化 | 已有 cron、wakeups、webhooks 等 | 已实现 cron、heartbeat、后台任务、重启与取消 |
| 会话与状态 | 已有 sessions、presence、config、cron 等网关状态面 | 已实现 append-only session、memory、workflow、provider health 持久化 |
| 工具能力 | 已有 browser、canvas、nodes、cron、sessions、聊天平台动作等工具 | 已实现 filesystem、shell、web、message、spawn、cron、MCP |
| 模型与容错 | 已有 models、model failover、retry policy | 已实现 provider 路由、模型回退、健康缓存、重试逻辑 |

## 多 Agent 架构

`fubot` 的核心不是单代理循环，而是一个正式的多 agent 运行时：

- `Coordinator` 负责识别任务类型、选择执行器并创建 workflow
- `Executors` 负责实际执行任务，每个执行器都有独立角色和工具权限
- `Router` 负责按任务和模型健康状态选择模型与 Provider
- `WorkflowStore` 负责保存 workflow、task、assignment、execution log、shared board 和 provider health

默认内置 5 个执行角色：

- `generalist`：通用任务和用户沟通
- `builder`：编码、修改、重构
- `researcher`：搜索、信息整理、证据收集
- `verifier`：测试、校验、代码审查
- `operator`：定时任务、运维和执行型操作

这套结构的重点是把“调度”和“执行”拆开，让不同类型任务在不同角色下运行，并且把状态、日志和路由决策保留下来。

## 安装方法

### 方式一：本地安装（推荐）

要求：

- Python 3.11+
- `pip`

安装：

```bash
python3 -m pip install -e .
```

如果你需要开发依赖：

```bash
python3 -m pip install -e .[dev]
```

初始化配置与工作区：

```bash
fubot onboard
```

然后编辑 `~/.fubot/config.json`，至少填写 `llm` 配置。一个最小可运行的 OpenAI 兼容示例：

```json
{
  "llm": {
    "provider": "custom",
    "baseUrl": "https://your-openai-compatible-endpoint/v1",
    "apiKey": "YOUR_API_KEY",
    "modelId": "YOUR_MODEL_ID"
  }
}
```

### 方式二：使用仓库内的运行时配置

仓库中带有一个可直接传入的运行时配置文件：

```bash
fubot agent -c runtime/config.json -m "你好"
```

如果你使用这份配置，建议先检查并替换里面的模型连接信息，再投入正式使用。

### 方式三：Docker

```bash
docker build -t fubot .
docker compose up fubot-gateway
```

## 使用方法

### 1. 单次执行

```bash
fubot agent -m "帮我总结一下当前目录的项目结构"
```

### 2. 交互模式

```bash
fubot agent
```

### 3. 指定配置文件

```bash
fubot agent -c runtime/config.json
```

### 4. 启动 Gateway

```bash
fubot gateway -c runtime/config.json
```

Gateway 适合接入聊天通道、定时任务、Heartbeat 和持续运行场景。

### 5. 查看状态

```bash
fubot status
fubot channels status
```

### 6. Provider 登录

适用于 OAuth 类 Provider：

```bash
fubot provider login openai-codex
```

### 7. 常见使用路径

- 想本地对话：`fubot onboard` -> 配置 `~/.fubot/config.json` -> `fubot agent`
- 想做自动运行或多通道接入：配置通道 -> `fubot gateway`
- 想做代码、搜索、定时任务混合场景：直接使用默认多代理运行时即可
