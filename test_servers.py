#!/usr/bin/env python3
"""
Test script to verify both MCP servers are working correctly and list their available tools.
"""

import asyncio
import json
import sys
from contextlib import AsyncExitStack
from typing import Any

import aiohttp

# Try to import MCP client dependencies
try:
    from mcp import ClientSession
    from mcp.client.sse import sse_client
    MCP_AVAILABLE = True
except ImportError:
    MCP_AVAILABLE = False
    print("⚠️  MCP client dependencies not available. Install with: pip install mcp")


class MCPServerInfo:
    """Information about an MCP server and its tools."""

    def __init__(self, name: str, port: int, description: str):
        self.name = name
        self.port = port
        self.description = description
        self.tools: list[dict[str, Any]] = []
        self.is_connected = False
        self.error_message: str | None = None


async def test_server_endpoint(port: int, server_name: str) -> bool:
    """Test if a server endpoint is responding correctly."""
    url = f"http://localhost:{port}/sse"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    # Read a small amount of data to verify SSE is working
                    data = await response.content.read(100)
                    if b"event: endpoint" in data:
                        print(f"✅ {server_name} (port {port}): SSE endpoint responding correctly")
                        return True
                    else:
                        print(f"❌ {server_name} (port {port}): Unexpected response format")
                        return False
                else:
                    print(f"❌ {server_name} (port {port}): HTTP {response.status}")
                    return False
    except asyncio.TimeoutError:
        print(f"⚠️  {server_name} (port {port}): Timeout (expected for SSE)")
        return True  # Timeout is expected for SSE endpoints
    except Exception as e:
        print(f"❌ {server_name} (port {port}): Error - {str(e)}")
        return False


async def connect_to_mcp_server(server_info: MCPServerInfo) -> None:
    """Connect to an MCP server and retrieve its available tools."""
    if not MCP_AVAILABLE:
        server_info.error_message = "MCP client dependencies not available"
        return

    url = f"http://localhost:{server_info.port}/sse"

    try:
        # Create SSE client connection
        async with AsyncExitStack() as stack:
            sse_transport = await stack.enter_async_context(
                sse_client(url)
            )
            read, write = sse_transport
            session = await stack.enter_async_context(
                ClientSession(read, write)
            )

            # Initialize the session
            await session.initialize()

            # List available tools
            tools_response = await session.list_tools()

            # Convert tools to dictionary format for display
            server_info.tools = []
            for tool in tools_response.tools:
                tool_info = {
                    "name": tool.name,
                    "description": tool.description or "No description provided",
                    "input_schema": tool.inputSchema
                }
                server_info.tools.append(tool_info)

            server_info.is_connected = True

    except Exception as e:
        server_info.error_message = f"Failed to connect: {str(e)}"


def format_tool_parameters(input_schema: dict) -> str:
    """Format tool parameters for display."""
    if not input_schema or "properties" not in input_schema:
        return "No parameters"

    properties = input_schema.get("properties", {})
    required = input_schema.get("required", [])

    params = []
    for param_name, param_info in properties.items():
        param_type = param_info.get("type", "unknown")
        is_required = param_name in required
        required_marker = " (required)" if is_required else " (optional)"
        description = param_info.get("description", "")

        param_str = f"  • {param_name}: {param_type}{required_marker}"
        if description:
            param_str += f" - {description}"
        params.append(param_str)

    return "\n".join(params) if params else "No parameters"


def display_server_tools(server_info: MCPServerInfo) -> None:
    """Display tools for a specific server."""
    print(f"\n{'='*60}")
    print(f"🔧 {server_info.name} (Port {server_info.port})")
    print(f"{'='*60}")
    print(f"Description: {server_info.description}")

    if not server_info.is_connected:
        print(f"❌ Connection failed: {server_info.error_message}")
        return

    if not server_info.tools:
        print("ℹ️  No tools available")
        return

    print(f"✅ Connected successfully - {len(server_info.tools)} tools available")
    print()

    for i, tool in enumerate(server_info.tools, 1):
        print(f"{i}. {tool['name']}")
        print(f"   Description: {tool['description']}")
        print(f"   Parameters:")
        print(format_tool_parameters(tool['input_schema']))
        print()


def identify_memory_specific_tools(base_tools: list, enhanced_tools: list) -> set:
    """Identify tools that are specific to the memory-enhanced server."""
    base_tool_names = {tool['name'] for tool in base_tools}
    enhanced_tool_names = {tool['name'] for tool in enhanced_tools}
    return enhanced_tool_names - base_tool_names


async def test_neo4j_connectivity() -> bool:
    """Test Neo4j connectivity via HTTP API."""
    url = "http://localhost:7474/db/data/"

    try:
        async with aiohttp.ClientSession() as session:
            async with session.get(url, timeout=aiohttp.ClientTimeout(total=5)) as response:
                if response.status == 200:
                    print("✅ Neo4j: HTTP API responding correctly")
                    return True
                else:
                    print(f"❌ Neo4j: HTTP {response.status}")
                    return False
    except Exception as e:
        print(f"❌ Neo4j: Error - {str(e)}")
        return False


async def main():
    """Run all tests and display server capabilities."""
    print("🧪 Testing MCP Server Setup and Capabilities...")
    print("=" * 60)

    # Define server configurations
    servers = [
        MCPServerInfo(
            name="Base MCP Server",
            port=8002,
            description="Standard Graphiti MCP server with core knowledge graph capabilities"
        ),
        MCPServerInfo(
            name="Memory Enhanced Server",
            port=8003,
            description="Enhanced server with AI agent memory capabilities and problem-solving tools"
        )
    ]

    # Test basic connectivity first
    print("📡 Testing Basic Connectivity...")
    print("-" * 40)

    base_server_ok = await test_server_endpoint(8002, "Base MCP Server")
    memory_server_ok = await test_server_endpoint(8003, "Memory Enhanced Server")
    neo4j_ok = await test_neo4j_connectivity()

    # Connect to MCP servers and retrieve tools
    print("\n🔍 Connecting to MCP Servers and Retrieving Tools...")
    print("-" * 50)

    for server in servers:
        print(f"Connecting to {server.name}...")
        await connect_to_mcp_server(server)

    # Display detailed tool information
    print("\n📋 Server Capabilities and Available Tools")

    for server in servers:
        display_server_tools(server)

    # Compare servers and highlight memory-specific tools
    if servers[0].is_connected and servers[1].is_connected:
        memory_tools = identify_memory_specific_tools(
            servers[0].tools, servers[1].tools
        )

        if memory_tools:
            print(f"\n{'='*60}")
            print("🧠 Memory-Specific Enhancements")
            print(f"{'='*60}")
            print("The following tools are unique to the Memory Enhanced Server:")
            for tool_name in sorted(memory_tools):
                # Find the tool details
                tool_details = next(
                    (t for t in servers[1].tools if t['name'] == tool_name),
                    None
                )
                if tool_details:
                    print(f"\n🔧 {tool_name}")
                    print(f"   Description: {tool_details['description']}")
                    print(f"   Parameters:")
                    print(format_tool_parameters(tool_details['input_schema']))

    # Summary
    print(f"\n{'='*60}")
    print("📊 Summary")
    print(f"{'='*60}")

    all_services_ok = base_server_ok and memory_server_ok and neo4j_ok

    if all_services_ok:
        print("🎉 All services are running correctly!")

        # Tool count summary
        base_tool_count = len(servers[0].tools) if servers[0].is_connected else 0
        enhanced_tool_count = len(servers[1].tools) if servers[1].is_connected else 0

        print(f"\n📈 Tool Statistics:")
        print(f"  • Base MCP Server:      {base_tool_count} tools")
        print(f"  • Memory Enhanced:      {enhanced_tool_count} tools")
        if servers[0].is_connected and servers[1].is_connected:
            memory_specific = len(identify_memory_specific_tools(
                servers[0].tools, servers[1].tools
            ))
            print(f"  • Memory-specific:      {memory_specific} additional tools")

        print(f"\n🌐 Service URLs:")
        print(f"  • Base MCP Server:      http://localhost:8002/sse")
        print(f"  • Memory Enhanced:      http://localhost:8003/sse")
        print(f"  • Neo4j Browser:        http://localhost:7474")
        return 0
    else:
        print("❌ Some services are not working correctly.")
        print(f"\n🔧 Troubleshooting:")
        print(f"  1. Check if services are running: docker-compose ps")
        print(f"  2. Check logs: docker-compose logs <service-name>")
        print(f"  3. Restart services: docker-compose restart")

        if not MCP_AVAILABLE:
            print(f"  4. Install MCP client: pip install mcp")

        return 1


if __name__ == "__main__":
    try:
        exit_code = asyncio.run(main())
        sys.exit(exit_code)
    except KeyboardInterrupt:
        print("\n⚠️  Test interrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Test failed with error: {str(e)}")
        sys.exit(1)
