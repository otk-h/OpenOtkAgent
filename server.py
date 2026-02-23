# Please install mcp first: `pip install mcp`
from mcp.server.fastmcp import FastMCP
# import psutil
# import platform
# import sys
from datetime import datetime
from rag_engine import rag_instance
import os

mcp = FastMCP("SystemMonitor")

# ---------- Local Tools ----------

@mcp.tool()
def get_system_time():
    return datetime.now().strftime("%Y-%m-%d %H:%M:%S")

@mcp.tool()
def list_files(directory=".") -> str:
    try:
        files = os.listdir(directory)
        return f"DIR {directory}: " + ", ".join(files)
    except Exception as e:
        return f"ERROR: {str(e)}"

@mcp.tool()
def read_file(filename: str) -> str:
    try:
        with open(filename, 'r', encoding='utf-8') as file:
            return file.read()
    except Exception as e:
        return f"ERROR: {str(e)}"

@mcp.tool()
def write_file(filename: str, content: str) -> str:
    try:
        with open(filename, 'w', encoding='utf-8') as file:
            file.write(content)
        return f"SUCCESS: File {filename} has been written."
    except Exception as e:
        return f"ERROR: {str(e)}"

# ---------- RAG Tools ----------

@mcp.tool()
def search_docs(query: str) -> str:
    return rag_instance.query(query)

# @mcp.tool()
# def get_system_stats():
#     """获取当前系统的 CPU 和内存使用率"""
#     # interval=1 会阻塞 1 秒，对演示友好
#     cpu_usage = psutil.cpu_percent(interval=0.1)
#     memory = psutil.virtual_memory()
#     return f"OS: {platform.system()}, CPU: {cpu_usage}%, Memory: {memory.percent}%"

if __name__ == "__main__":
    mcp.run()
