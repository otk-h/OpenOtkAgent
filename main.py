# Please install OpenAI SDK first: `pip3 install openai`
import os
import openai
import json
from datetime import datetime

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

tools = [
    {
        "type": "function",
        "function": {
            "name": "get_current_time",
            "description": "Get the current date and time",
            "parameters": {"type": "object", "properties": {}}
        }
    },
    {
        "type": "function",
        "function": {
            "name": "list_files",
            "description": "list files' name in certain directory",
            "parameters": {
                "type": "object",
                "properties": {
                    "directory": {
                        "type": "string",
                        "description": "directory path, default is current directory"
                    }
                }
            }
        }
    },
]

def run_agent():
    messages = [
        # system   : global instructions
        # user     : user input
        # assistant: model output
        {"role": "system", "content": "You are a helpful assistant"},
    ]
    
    print("program started")
    
    while True:
        # Get user input
        user_input = input("User (input 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        messages.append({"role": "user", "content": user_input})
        
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                tools=tools,
                stream=False
            )

            response_message = response.choices[0].message
            # 1. if the model is calling a tool
            if response_message.tool_calls:
                messages.append(response_message)
                for tool_call in response_message.tool_calls:
                    function_name = tool_call.function.name
                    function_args = json.loads(tool_call.function.arguments)
                    print(f"system: Agent is calling tool: {function_name} with arguments: {function_args}")
                    # 2. execute the tool function and get the result
                    if function_name == "get_current_time":
                        result = get_current_time()
                    elif function_name == "list_files":
                        directory = function_args.get("directory", ".")
                        result = list_files(directory)
                    else:
                        result = f"ERROR: Unknown function: {function_name}"
                    print(f"system: Tool response: {result}")
                    # 3. append the tool response
                    messages.append({
                        "role": "tool",
                        "tool_call_id": tool_call.id,
                        "name": function_name,
                        "content": result
                    })
                # 4. finish call tolls, call model with the tool response
                second_response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=messages,
                )
                final_output = second_response.choices[0].message.content
                print(f"Assistant: {final_output}")
                messages.append({"role": "assistant", "content": final_output})
            else:
                # the model is not calling any tool
                final_output = response_message.content
                print(f"Assistant: {final_output}")
                messages.append({"role": "assistant", "content": final_output})
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    run_agent()
