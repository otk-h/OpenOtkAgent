# OpenOtkAgent

通过一个小玩具来了解 Agent

## Step 1: Hello World

实现最基础 API 调用  
通过 chat_history 记录对话历史  

```py
chat_history = [
    # system   : global instructions
    # user     : user input
    # assistant: model output
    {"role": "system", "content": SYSTEM_PROMPT},
]
```

对每个用户输入或每轮思考，将所有 chat_history 重新输入给大模型，实现连续对话  

## Step 2: Function Calling

提前编码可以让大模型调用的工具，如文件读写等。在 ``SYSTEM_PROMPT`` 中严格定义输出格式，根据大模型输出结果调用相应工具  

对于函数，参数等，注意使用 JSON 格式化  

## Step 3: ReAct

在 ``SYSTEM_PROMPT`` 中加入 ``Thought-Action-Observation`` 循环  

## Step 4: RAG

通过 `chromadb` 实现检索增强生成 (RAG: Retrieval-Augmented Generation)  

1. 提前在 `knowledge` 目录下放入可以被检索的文本文件
2. 将文本 (分段) 转换为高维向量 (Embedding)，建立向量数据库
3. 将用户输入转换为向量，匹配数据库中距离最近的片段

需要提前运行 `tool_rag.py` 加载文档，后续每次运行 `main.py` 都不用重新加载  

## Step 5: MCP

将工具封装在独立的 `MCP Server` 中，Agent 链接 Server 后自动发现并使用工具  

1. MCP Host: 与 LLM 交互，管理与 MCP Server 的链接
2. MCP Server: 独立进程，暴露数据，函数，模板
3. Transport: 用 JSON 进行 Host 和 Server 间通信

对过去代码的重构：  

1. `prompt.txt` 变为模板，启动 agent 时动态生成
2. 将 RAG 功能封装，与其他工具一起放入 MCP 工具
