# Please install OpenAI SDK first: `pip3 install openai`
import os
import openai
import json
import re

from tool_normal import *
from tool_rag import load_documents, search_docs

client = openai.OpenAI(
    # Set your API key in the environment variable `DEEPSEEK_API_KEY` before running this code
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

TOOLS = {
    "get_current_time": get_current_time,
    "list_files": list_files,
    "read_file": read_file,
    "write_file": write_file,
    "search_docs": search_docs,
}

def run_agent():
    print("Welcome to OpenOtkAgent! Type 'exit' to quit.")
    with open("prompt.txt", 'r') as file:
        SYSTEM_PROMPT=file.read()
    
    chat_history = [
        # system   : global instructions
        # user     : user input
        # assistant: model output
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    while True:
        # Get user input
        user_input = input("User: ")
        if user_input.lower() == 'exit':
            break
        chat_history.append({"role": "user", "content": user_input})
        
        for i in range(5):
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=chat_history,
                stop=["Observation:"]
            )
            content = response.choices[0].message.content
            chat_history.append({"role": "assistant", "content": content})
            print(f"--- Reasoning Turns {i+1} ---")
            print(content)
            
            if "Final Answer:" in content:
                break

            action_match = re.search(r"Action:\s*(\w+)\[([\s\S]*?)\]", content)
            if action_match:
                tool_name = action_match.group(1)
                tool_args = action_match.group(2)
                
                print(f"System call tool: {tool_name}, args: {tool_args}")
                try:
                    args = json.loads(tool_args) if tool_args.strip() else {}
                
                    if tool_name in TOOLS:
                        observation = TOOLS[tool_name](**args)
                    else:
                        observation = f"Error: Unknown tool {tool_name}"
                except json.JSONDecodeError:
                    observation = "Error: Invalid JSON format in Action parameters."
                except TypeError as e:
                    observation = f"Error: Parameter mismatch for {tool_name}. {str(e)}"
                    
                print(f"Observation: {observation}")
                chat_history.append({"role": "user", "content": f"Observation: {observation}"})
            else:
                chat_history.append({"role": "user", "content": "Observation: Please continue to give Action or Final Answer."})
                break
    print("Agent session ended.")

if __name__ == "__main__":
    run_agent()
