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

TBD

## Step 5: MCP

TBD
