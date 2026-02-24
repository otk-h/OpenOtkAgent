# Please install OpenAI SDK first: `pip3 install openai`
# Please install mcp first: `pip install mcp`
import asyncio
import sys
import os
import operator
from typing import Annotated, List, Union, Optional
from typing_extensions import TypedDict
from datetime import datetime

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import BaseMessage, SystemMessage, ToolMessage, HumanMessage, AIMessage
from langgraph.checkpoint.memory import MemorySaver

from pydantic import BaseModel, Field

with open("prompt/prompt_Planner.txt", 'r') as file:
    PROMPT_PLANNER = file.read()
with open("prompt/prompt_RePlanner.txt", 'r') as file:
    PROMPT_REPLANNER = file.read()
with open("prompt/prompt_Executor.txt", 'r') as file:
    PROMPT_EXECUTOR = file.read()

MODEL_NAME = "deepseek-chat"
API_KEY = os.environ.get('DEEPSEEK_API_KEY')
BASE_URL = "https://api.deepseek.com"

MCP_SERVER = "mcp_server.py"

EXECUTOR_LOOP = 5
EXECUTOR_TIMELIMIT = 60.0

# ---------- Schema ----------

class Plan(BaseModel):
    steps: List[str] = Field(
        description="different steps to follow, should be in sorted order"
    )
    
class Response(BaseModel):
    response: str

class Act(BaseModel):
    action: Union[Response, Plan] = Field(
        description="Action to perform. If you want to respond to user, use Response. If you need to further use tools to get the answer, use Plan."
    )

# ---------- State ----------

class PlanExecuteState(TypedDict):
    # Annotated[..., operator.add]: add new message to history
    global_history: Annotated[List[BaseMessage], operator.add]  # for global agent
    internal_history: List[BaseMessage]                         # for executor sub-agent
    
    input: str
    plan: List[str]
    past_steps: Annotated[List[str], operator.add]
    response: Optional[str]

# ---------- Node ----------

class Agent:
    def __init__(self, mcp_session: ClientSession):
        self.mcp_session = mcp_session
        self.model = ChatOpenAI(
            model=MODEL_NAME,
            api_key=API_KEY,
            base_url=BASE_URL,
            temperature=0,
        )
        # self.model_planner = self.model.with_structured_output(Plan)
        # self.model_replanner = self.model.with_structured_output(Act)
        self.model_planner = self.model.with_structured_output(Act, method="function_calling")
        self.model_replanner = self.model.with_structured_output(Act, method="function_calling")
        self.model_executor = None
        
        self.tools = []
    
    async def initialize_tools(self):
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
        # 4. bind tools to the model
        self.model_executor = self.model.bind_tools(self.tools)

    # ---------- Node: planner ----------
    async def planner(self, state: PlanExecuteState):
        print(f"[Planner] Generating plan for: {state['input']}")
        prompt_template = ChatPromptTemplate.from_template(PROMPT_PLANNER)
        prompt_str = prompt_template.format(
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            tools = self.tools,
            query = state["input"],
        )
        prompt = state.get("global_history", []) + [HumanMessage(content=prompt_str)]
        decision = await self.model_planner.ainvoke(prompt)
        
        if isinstance(decision.action, Response):
            print(f"   - Simple query detected, responding directly.")
            return {"response": decision.action.response}
        else:
            print(f"   - Updated plan: {decision.action.steps}")
            return {"plan": decision.action.steps}

    # ---------- Node: replanner ----------
    async def replanner(self, state: PlanExecuteState):
        print(f"\n[Replanner] Reviewing progress...")
        prompt_template = ChatPromptTemplate.from_template(PROMPT_REPLANNER)
        prompt_str = prompt_template.format(
            date = datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            query = state["input"],
            completed_steps = state['past_steps'],
            remaining_steps = state['plan'],
        )
        prompt = state.get("global_history", []) + [HumanMessage(content=prompt_str)]
        decision = await self.model_replanner.ainvoke(prompt)
        
        if isinstance(decision.action, Response):
            return {"response": decision.action.response}
        else:
            print(f"   - Updated plan: {decision.action.steps}")
            return {"plan": decision.action.steps}

    # ---------- Node: executor ----------
    async def executor(self, state: PlanExecuteState):
        current_step = state["plan"][0]
        print(f"\n[Executor] executing step: {current_step}")
        prompt_template = ChatPromptTemplate.from_template(PROMPT_EXECUTOR)
        prompt_str = prompt_template.format(
            current_step = current_step,
            completed_steps = "\n".join(state["past_steps"]),
        )
        
        internal_history = state.get("internal_history", [])
        if not internal_history:
            internal_history = [SystemMessage(content=prompt_str)]
        
        for _ in range(EXECUTOR_LOOP):
            response = await self.model_executor.ainvoke(internal_history)
            internal_history.append(response)
        
            if response.tool_calls:
                for tool_call in response.tool_calls:
                    tool_name = tool_call["name"]
                    tool_args = tool_call["args"]
                    print(f"   - Tool Call: {tool_name}({tool_args})")
                    
                    try:
                        tool_result = await asyncio.wait_for(
                            self.mcp_session.call_tool(tool_name, arguments=tool_args),
                            timeout=EXECUTOR_TIMELIMIT
                        )
                        observation = tool_result.content[0].text
                    except asyncio.TimeoutError:
                        observation = "ERROR: tool calling reach the time limit. Retry or try another tool"
                    except Exception as e:
                        observation = f"ERROR: str{e}"
                        
                    print(f"   - Observation: {observation}")
                    internal_history.append(ToolMessage(content=observation, tool_call_id=tool_call["id"]))
                continue
            else:
                step_summary = f"step: {current_step} | result: {response.content}"
                break
        
        if not step_summary:
            step_summary = "Executor loop reached limit without a text summary."
        
        return {
            "past_steps": [f"Step: {current_step} | Result: {step_summary}"],
            "plan": state["plan"][1:],
            "internal_history": [],
        }
    
    def build_graph(self):
        workflow = StateGraph(PlanExecuteState)
    
        workflow.add_node("planner", self.planner)
        workflow.add_node("executor", self.executor)
        workflow.add_node("replanner", self.replanner)
        workflow.set_entry_point("planner")
        
        workflow.add_edge("executor", "replanner")
        
        def after_plan(state: PlanExecuteState):
            if state.get("response") and not state.get("plan"):
                return "end"
            return "execute"
        
        def after_replan(state: PlanExecuteState):
            if state.get("response"):
                return "end"
            return "execute"
        
        workflow.add_conditional_edges(
            "planner",
            after_plan,
            {
                "execute": "executor",
                "end": END
            }
        )
        
        workflow.add_conditional_edges(
            "replanner",
            after_replan,
            {
                "execute": "executor",
                "end": END
            }
        )
        
        memory = MemorySaver()
        return workflow.compile(checkpointer=memory)

async def main():
    server_params = StdioServerParameters(
        command=sys.executable,
        args=[MCP_SERVER],
        env=os.environ.copy()
    )
    
    async with stdio_client(server_params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            agent = Agent(session)
            await agent.initialize_tools()
            graph = agent.build_graph()
            
            config = {
                "configurable": {
                    "thread_id": "114514"
                },
                "recursion_limit": 50
            }
            
            print("==========================")
            print("--- OpenOtkAgent Start ---")
            print("==========================")
            
            while True:
                user_input = input("User: ")
                if user_input.lower() in ['exit', 'quit', 'q']: break
    
                state = {
                    "global_history": [HumanMessage(content=user_input)],
                    "internal_history": [],
                    "input": user_input,
                    "plan": [],
                    "past_steps": [],
                    "response": None,
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
                            print(f"Assistant: {output['response']}")

if __name__ == "__main__":
    asyncio.run(main())
