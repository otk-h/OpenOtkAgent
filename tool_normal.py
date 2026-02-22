import os
from datetime import datetime

def get_current_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

def list_files(directory="."):
    try:
        files = os.listdir(directory)
        return f"DIR {directory}: " + ", ".join(files)
    except Exception as e:
        return f"ERROR: {str(e)}"
    
def read_file(filename):
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"ERROR: {str(e)}"

def write_file(filename, content):
    try:
        with open(filename, 'w', encoding='utf-8') as f:
            f.write(content)
        return f"SUCCESS: File {filename} has been written."
    except Exception as e:
        return f"ERROR: {str(e)}"