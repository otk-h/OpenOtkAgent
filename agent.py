# Please install OpenAI SDK first: `pip3 install openai`
# Please install mcp first: `pip install mcp`
import openai
import asyncio
import sys
import json
import os
import re
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

class Agent:
    def __init__(self):
        self.client = openai.OpenAI(
            # Set your API key in the environment variable `DEEPSEEK_API_KEY` before running this code
            api_key=os.environ.get('DEEPSEEK_API_KEY'),
            base_url="https://api.deepseek.com"
        )
        self.toolbox = {}
        # Set True to print reasoning process
        self.debug = True
    
    async def run(self, server_script: str):
        print("Welcome to OpenOtkAgent! Type 'exit' to quit.")
        
        # 1. start MCP server
        server_params = StdioServerParameters(
            command=sys.executable,
            args=[server_script],
            env=os.environ.copy()
        )
        
        async with stdio_client(server_params) as (read, write):
            async with ClientSession(read, write) as session:
                await session.initialize()
                # 2. dynamically load tools from MCP server
                mcp_tools = await session.list_tools()
                for tool in mcp_tools.tools:
                    self.toolbox[tool.name] = tool
                
                print(f"Loaded tools: {list(self.toolbox.keys())}")
                
                # 3. dynamically build system prompt with tool descriptions
                system_prompt = self.build_system_prompt()
                chat_history = [
                    # system   : global instructions
                    # user     : user input
                    # assistant: model output
                    {"role": "system", "content": system_prompt},
                ]
                
                # 4. main interaction loop
                while True:
                    # Get user input
                    user_input = input("User: ")
                    if user_input.lower() == 'exit': break
                    chat_history.append({"role": "user", "content": user_input})
                        
                    for i in range(5):
                        response = self.client.chat.completions.create(
                            model="deepseek-chat",
                            messages=chat_history,
                            stop=["Observation:"],
                        )
                        content = response.choices[0].message.content
                        chat_history.append({"role": "assistant", "content": content})
                        
                        if self.debug:
                            print(f"--- Reasoning Turns {i+1} ---")
                            print(content)
                        
                        if "Final Answer:" in content: break

                        action_match = re.search(r"Action:\s*(\w+)\[([\s\S]*?)\]", content)
                        if action_match:
                            tool_name = action_match.group(1)
                            tool_args = action_match.group(2)

                            if self.debug:
                                print(f"System call tool: {tool_name}, args: {tool_args}")

                            try:
                                args = json.loads(tool_args) if tool_args.strip() else {}
                                result = await session.call_tool(tool_name, arguments=args)
                                observation = result.content[0].text
                            except json.JSONDecodeError:
                                observation = "ERROR: Invalid JSON format in Action parameters."
                            except TypeError as e:
                                observation = f"ERROR: Parameter mismatch for {tool_name}. {str(e)}"
                            except Exception as e:
                                observation = f"ERROR: {str(e)}"
                            
                            if self.debug:
                                print(f"Observation: {observation}")
                              
                            chat_history.append({"role": "user", "content": f"Observation: {observation}"})
                        else:
                            chat_history.append({"role": "user", "content": "Observation: Please continue to give Action or Final Answer."})
                            break
                print("Agent session ended.")
            
    def build_system_prompt(self):
        with open("prompt.txt", 'r') as file:
            template = file.read()

        tools_desc = ""
        for name, tool in self.toolbox.items():
            tools_desc += f"- {name}[{json.dumps(tool.inputSchema.get('properties', {}))}]: {tool.description}\n"

        return template.replace("{{TOOLS_LIST}}", tools_desc)

if __name__ == "__main__":
    agent = Agent()
    asyncio.run(agent.run("server.py"))
