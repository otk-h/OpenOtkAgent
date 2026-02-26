# Please install mcp first: `pip install mcp`
from mcp.server.fastmcp import FastMCP
from datetime import datetime
import asyncio
import os

from rag.rag_engine import rag_instance

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
async def search_docs(query: str) -> str:
    result = await asyncio.to_thread(rag_instance.query, query)
    return result

if __name__ == "__main__":
    mcp.run()
