from mcp.client.sse import sse_client
from mcp import ClientSession
import asyncio

async def main():
    # Connect to the MCP server using SSE with the correct URL format
    async with sse_client("http://localhost:8000/sse") as (read_stream, write_stream):
        # Create a session using the client streams
        async with ClientSession(read_stream, write_stream) as session:
            # Initialize the connection
            await session.initialize()
            
            # Test the retrieve tool
            print("\nTesting retrieve tool...")
            retrieve_result = await session.call_tool(
                "retrieve", 
                {"query": "Who is Lionel Messi?", "top_k": 3}
            )
            print("Retrieve result:", retrieve_result)
            
            # Test the rag tool
            print("\nTesting rag tool...")
            rag_result = await session.call_tool(
                "rag",
                {"query": "Who is Lionel Messi?"}
            )
            print("RAG result:", rag_result)

if __name__ == "__main__":
    asyncio.run(main()) 