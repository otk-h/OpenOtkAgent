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

需要提前运行 `rag_loader.py` 加载文档，后续每次运行 `agent.py` 都不用重新加载  

## Step 5: MCP

将工具封装在独立的 `MCP Server` 中，Agent 链接 Server 后自动发现并使用工具  

1. MCP Host: 与 LLM 交互，管理与 MCP Server 的链接
2. MCP Server: 独立进程，暴露数据，函数，模板
3. Transport: 用 JSON 进行 Host 和 Server 间通信

对过去代码的重构：  

1. `prompt.txt` 变为模板，启动 agent 时动态生成
2. 将 RAG 功能封装，与其他工具一起放入 MCP 工具，用 `@mcp.tool()` 修饰

## Step 6: LangGraph (ReAct -> Plan-Execute)

用状态机的思维管理 Agent 的逻辑，构建有状态，多角色，复杂循环  

>可以在关键节点设置断点，等待人工审核

1. Node: 一个 python 函数，如调用 LLM，执行工具等
2. Edge: 条件 / 无条件，定义下一步逻辑
3. State: 一个共享数据结构，记录当前工作流状态

对过去代码的重构:  

1. 引入 LangGraph 抽象层，不需要 openai
2. prompt 不再包含工具清单，只包含身份，原则，策略

实现模板参考 [LangGraph 教程](https://github.langchain.ac.cn/langgraph/tutorials/plan-and-execute/plan-and-execute/)

### 6.1 动态获取 MCP 工具并渲染

```py
# 1. get MCP tools
mcp_tools = (await self.mcp_session.list_tools()).tools
# 2. render tools for LLM
self.tools = [{
    "type": "function",
    "function": {
        "name": tool.name,
        "description": tool.description,
        "parameters": tool.inputSchema,
    }
} for tool in mcp_tools]
# 3. bind tools to the model
self.model_executor = self.model.bind_tools(self.tools)
```

### 6.2 构建流程图

```py
workflow = StateGraph(PlanExecuteState)

workflow.add_node("planner", self.planner)
workflow.add_node("executor", self.executor)
workflow.add_node("replanner", self.replanner)
workflow.set_entry_point("planner")

workflow.add_edge("planner", "executor")
workflow.add_edge("executor", "replanner")

def should_continue(state: PlanExecuteState):
    if state.get("response"):
        return "end"
    return "continue"

workflow.add_conditional_edges(
    "replanner",
    should_continue,
    {
        "continue": "executor",
        "end": END
    }
)

return workflow.compile()
```

### 6.3 编码流程图中每个节点逻辑

```py
# ---------- Node: planner ----------
async def planner(self, state: PlanExecuteState):
    prompt = "..."
    response = await self.model_planner.ainvoke([SystemMessage(content=prompt)])
    return {"plan": response.steps}

# ---------- Node: replanner ----------
async def replanner(self, state: PlanExecuteState):
    prompt = "..."
    decision = await self.model_replanner.ainvoke([SystemMessage(content=prompt)])
    if isinstance(decision.action, Response):
        return {"response": decision.action.response}
    else:
        return {"plan": decision.action.steps}
```

调用 MCP 工具可用以下代码

```py
for tool_call in response.tool_calls:
    tool_name = tool_call["name"]
    tool_args = tool_call["args"]
    print(f"   - Tool Call: {tool_name}({tool_args})")
    
    tool_result = await self.mcp_session.call_tool(tool_name, arguments=tool_args)
    observation = tool_result.content[0].text
```

### 6.4 executor tip

1. 将 executor 设计为一个 sub-agent，否则会在 replanner 和 executor 中无限循环，无法理解当前 step 进度  
2. 善用 asyncio 防止 executor 执行时间过长

## Step 7: CheckPoint

1. 不再手动维护 state 的 history，使用 LangGraph 提供的持久化机制  
2. 为每个 session 分配一个 thread_id，自动保存每一轮 state  
3. 下次用相同 thread_id 调用图时，自动加载 history 和新 user_input

```py
# build_graph
memory = MemorySaver()
return workflow.compile(checkpointer=memory)
```

```py
# main
config = {
    "configurable": {
        "thread_id": "114514"
    },
    "recursion_limit": 50
}
async for event in graph.astream(state, config=config):
    for _, output in event.items():
        if "response" in output and output["response"]:
            await graph.aupdate_state(
                config,
                {
                    "global_history": [AIMessage(content=output["response"])]
                }
            )
```

## Step 8: Skill

TBD
