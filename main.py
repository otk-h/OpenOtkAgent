# Please install OpenAI SDK first: `pip3 install openai`
import os
import openai

client = openai.OpenAI(
    # Set your API key in the environment variable `DEEPSEEK_API_KEY` before running this code
    api_key=os.environ.get('DEEPSEEK_API_KEY'),
    base_url="https://api.deepseek.com"
)

def simple_agent():
    messages = [
        # system   : global instructions
        # user     : user input
        # assistant: model output
        {"role": "system", "content": "You are a helpful assistant"},
    ]
    
    print("program started")
    
    while True:
        user_input = input("User (input 'exit' to quit): ")
        if user_input.lower() == 'exit':
            print("Exiting the program.")
            break
        messages.append({"role": "user", "content": user_input})
        try:
            response = client.chat.completions.create(
                model="deepseek-chat",
                messages=messages,
                stream=False
            )
            answer = response.choices[0].message.content
            print("Assistant:", answer)
            messages.append({"role": "assistant", "content": answer})
        except Exception as e:
            print(f"An error occurred: {e}")

if __name__ == "__main__":
    simple_agent()
