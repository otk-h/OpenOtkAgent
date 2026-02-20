# Please install OpenAI SDK first: `pip3 install openai`
import os
import openai
import json
import re
from datetime import datetime

from sympy import content

client = openai.OpenAI(
    # Set your API key in the environment variable `DEEPSEEK_API_KEY` before running this code
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def list_files(directory="."):
    try:
        files = os.listdir(directory)
        return f"DIR {directory}: " + ", ".join(files)
    except Exception as e:
        return f"ERROR: {str(e)}"

TOOLS = {
    "get_current_time": get_current_time,
    "list_files": list_files,
}

with open("prompt.txt", 'r') as file:
    SYSTEM_PROMPT=file.read()

def run_agent():
    chat_history = [
        # system   : global instructions
        # user     : user input
        # assistant: model output
        {"role": "system", "content": SYSTEM_PROMPT},
    ]
    
    print("program started")
    
    while True:
        # Get user input
        user_input = input("User (input 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        chat_history.append({"role": "user", "content": user_input})
        
        for i in range(5):
            try:
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

                action_match = re.search(r"Action:\s*(\w+)\[(.*)\]", content)
                if action_match:
                    tool_name = action_match.group(1)
                    tool_args = action_match.group(2)
                    if tool_name in TOOLS:
                        if tool_args.strip() == "":
                            observation = TOOLS[tool_name]()
                        else:
                            observation = TOOLS[tool_name](tool_args)
                        print(f"Observation: {observation}\n")
                        chat_history.append({"role": "user", "content": f"Observation: {observation}"})
                    else:
                        chat_history.append({"role": "user", "content": f"Observation: Error: Unknown tool {tool_name}"})
                else:
                    chat_history.append({"role": "user", "content": "Observation: Please continue to give Action or Final Answer."})
            except Exception as e:
                print(f"An error occurred: {e}")
                break

if __name__ == "__main__":
    run_agent()
