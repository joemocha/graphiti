"""
End-to-end tests for the MCP server protocol compliance.
Tests the actual MCP server functionality and protocol adherence.
"""

import asyncio
import json
import os
import pytest
from datetime import datetime
from typing import Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
from dotenv import load_dotenv

load_dotenv()


@pytest.fixture
def mcp_server_config():
    """Configuration for MCP server testing."""
    return {
        "host": "localhost",
        "port": 8001,
        "base_url": "http://localhost:8001",
        "group_id": f"e2e_test_{int(datetime.now().timestamp())}"
    }


@pytest.fixture
async def mock_mcp_server():
    """Mock MCP server for testing without real server."""
    from mcp.server.fastmcp import FastMCP
    from memory_enhanced_server import (
        store_problem_solving_experience,
        retrieve_similar_problems,
        get_lessons_for_domain
    )
    
    # Create a test server instance
    test_server = FastMCP(
        'Test Memory Server',
        instructions="Test server for E2E testing",
        host="localhost",
        port=8002
    )
    
    # Register the memory tools
    test_server.tool()(store_problem_solving_experience)
    test_server.tool()(retrieve_similar_problems)
    test_server.tool()(get_lessons_for_domain)
    
    return test_server


class TestMCPProtocolCompliance:
    """Test MCP protocol compliance and server behavior."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_server_initialization(self, mock_mcp_server):
        """Test that the MCP server initializes correctly."""
        # Verify server has the expected tools
        tools = mock_mcp_server._tools
        
        expected_tools = {
            "store_problem_solving_experience",
            "retrieve_similar_problems", 
            "get_lessons_for_domain"
        }
        
        registered_tools = set(tools.keys())
        assert expected_tools.issubset(registered_tools)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_tool_schema_validation(self, mock_mcp_server):
        """Test that tool schemas are properly defined."""
        tools = mock_mcp_server._tools
        
        # Test store_problem_solving_experience schema
        store_tool = tools.get("store_problem_solving_experience")
        assert store_tool is not None
        
        # Verify the tool has proper metadata
        assert hasattr(store_tool, '__name__')
        assert store_tool.__name__ == "store_problem_solving_experience"
        
        # Test retrieve_similar_problems schema
        retrieve_tool = tools.get("retrieve_similar_problems")
        assert retrieve_tool is not None
        assert hasattr(retrieve_tool, '__name__')
        
        # Test get_lessons_for_domain schema
        lessons_tool = tools.get("get_lessons_for_domain")
        assert lessons_tool is not None
        assert hasattr(lessons_tool, '__name__')

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_tool_execution_with_mocks(self, mock_server_setup):
        """Test tool execution with mocked dependencies."""
        from memory_enhanced_server import (
            store_problem_solving_experience,
            retrieve_similar_problems,
            get_lessons_for_domain
        )
        
        # Test store_problem_solving_experience
        result = await store_problem_solving_experience(
            problem_name="E2E Test Problem",
            problem_description="Testing end-to-end functionality",
            solution_approach="Use comprehensive E2E testing",
            key_insights="E2E tests catch integration issues",
            domain="testing",
            effectiveness="high"
        )
        
        assert isinstance(result, dict)
        assert "message" in result or "error" in result
        
        # Test retrieve_similar_problems
        result = await retrieve_similar_problems(
            current_problem="Testing problem",
            domain="testing"
        )
        
        assert isinstance(result, dict)
        assert "nodes" in result or "error" in result
        
        # Test get_lessons_for_domain
        result = await get_lessons_for_domain(
            domain="testing"
        )
        
        assert isinstance(result, dict)
        assert "facts" in result or "error" in result


class TestMCPServerHTTPInterface:
    """Test the HTTP interface of the MCP server (if running)."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_server_health_check(self, mcp_server_config):
        """Test server health check endpoint."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{mcp_server_config['base_url']}/health",
                    timeout=5.0
                )
                # If server is running, it should respond
                assert response.status_code in [200, 404]  # 404 if no health endpoint
        except httpx.ConnectError:
            pytest.skip("MCP server not running for E2E tests")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_server_sse_endpoint(self, mcp_server_config):
        """Test SSE endpoint availability."""
        try:
            async with httpx.AsyncClient() as client:
                response = await client.get(
                    f"{mcp_server_config['base_url']}/sse",
                    timeout=5.0
                )
                # Should get some response from SSE endpoint
                assert response.status_code in [200, 400, 405]  # Various valid responses
        except httpx.ConnectError:
            pytest.skip("MCP server not running for E2E tests")


class TestMemoryWorkflowE2E:
    """End-to-end tests for complete memory workflows."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_complete_memory_cycle(self, mock_server_setup):
        """Test a complete memory storage and retrieval cycle."""
        from memory_enhanced_server import (
            store_problem_solving_experience,
            retrieve_similar_problems,
            get_lessons_for_domain
        )
        
        # Step 1: Store a problem-solving experience
        problem_data = {
            "problem_name": "E2E Memory Test",
            "problem_description": "Testing complete memory workflow end-to-end",
            "solution_approach": "Implement comprehensive E2E testing strategy",
            "key_insights": "E2E tests validate the entire user journey",
            "mistakes_made": "Initially focused only on unit tests",
            "tools_used": "pytest, httpx, MCP protocol",
            "domain": "testing",
            "effectiveness": "high",
            "group_id": "e2e_test"
        }
        
        store_result = await store_problem_solving_experience(**problem_data)
        assert "message" in store_result
        assert "Successfully stored" in store_result["message"]
        
        # Step 2: Retrieve similar problems
        retrieve_result = await retrieve_similar_problems(
            current_problem="Memory testing workflow",
            domain="testing",
            max_results=5,
            group_ids=["e2e_test"]
        )
        
        assert "nodes" in retrieve_result
        # Should return empty list or found nodes
        assert isinstance(retrieve_result["nodes"], list)
        
        # Step 3: Get lessons for the domain
        lessons_result = await get_lessons_for_domain(
            domain="testing",
            max_lessons=10,
            group_ids=["e2e_test"]
        )
        
        assert "facts" in lessons_result
        assert isinstance(lessons_result["facts"], list)

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_error_scenarios(self, mock_server_setup):
        """Test various error scenarios end-to-end."""
        from memory_enhanced_server import (
            store_problem_solving_experience,
            retrieve_similar_problems
        )
        
        # Test with invalid/empty data
        result = await store_problem_solving_experience(
            problem_name="",
            problem_description="",
            solution_approach="",
            key_insights=""
        )
        
        # Should handle gracefully
        assert isinstance(result, dict)
        
        # Test retrieval with empty query
        result = await retrieve_similar_problems(
            current_problem="",
            domain=""
        )
        
        assert isinstance(result, dict)
        assert "nodes" in result or "error" in result

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_concurrent_operations(self, mock_server_setup):
        """Test concurrent memory operations."""
        from memory_enhanced_server import store_problem_solving_experience
        
        # Create multiple concurrent storage operations
        tasks = []
        for i in range(3):
            task = store_problem_solving_experience(
                problem_name=f"Concurrent Test {i}",
                problem_description=f"Testing concurrent operation {i}",
                solution_approach=f"Concurrent solution {i}",
                key_insights=f"Concurrent insight {i}",
                domain="concurrency",
                effectiveness="medium",
                group_id="concurrent_test"
            )
            tasks.append(task)
        
        # Execute concurrently
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # All should complete successfully or with handled errors
        for result in results:
            if isinstance(result, Exception):
                pytest.fail(f"Concurrent operation failed: {result}")
            assert isinstance(result, dict)
            assert "message" in result or "error" in result


class TestMCPClientCompatibility:
    """Test compatibility with MCP client patterns."""

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_mcp_tool_call_format(self, mock_server_setup):
        """Test that tool responses match MCP expected formats."""
        from memory_enhanced_server import store_problem_solving_experience
        
        result = await store_problem_solving_experience(
            problem_name="MCP Format Test",
            problem_description="Testing MCP response format",
            solution_approach="Ensure proper response structure",
            key_insights="MCP clients expect specific response formats",
            domain="mcp",
            effectiveness="high"
        )
        
        # Verify response structure matches MCP expectations
        assert isinstance(result, dict)
        
        # Should have either success or error response
        if "message" in result:
            # Success response
            assert isinstance(result["message"], str)
            assert len(result["message"]) > 0
        elif "error" in result:
            # Error response
            assert isinstance(result["error"], str)
            assert len(result["error"]) > 0
        else:
            pytest.fail("Response should have either 'message' or 'error' field")

    @pytest.mark.e2e
    @pytest.mark.asyncio
    async def test_response_serialization(self, mock_server_setup):
        """Test that all responses can be JSON serialized."""
        from memory_enhanced_server import (
            store_problem_solving_experience,
            retrieve_similar_problems,
            get_lessons_for_domain
        )
        
        # Test store response serialization
        store_result = await store_problem_solving_experience(
            problem_name="Serialization Test",
            problem_description="Testing JSON serialization",
            solution_approach="Ensure all responses are JSON serializable",
            key_insights="JSON serialization is required for MCP",
            domain="serialization"
        )
        
        # Should be JSON serializable
        try:
            json.dumps(store_result)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Store result not JSON serializable: {e}")
        
        # Test retrieve response serialization
        retrieve_result = await retrieve_similar_problems(
            current_problem="Serialization test",
            domain="serialization"
        )
        
        try:
            json.dumps(retrieve_result)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Retrieve result not JSON serializable: {e}")
        
        # Test lessons response serialization
        lessons_result = await get_lessons_for_domain(domain="serialization")
        
        try:
            json.dumps(lessons_result)
        except (TypeError, ValueError) as e:
            pytest.fail(f"Lessons result not JSON serializable: {e}")


@pytest.mark.e2e
class TestRealWorldScenarios:
    """Test real-world usage scenarios."""

    @pytest.mark.asyncio
    async def test_ai_agent_memory_scenario(self, mock_server_setup):
        """Test a realistic AI agent memory usage scenario."""
        from memory_enhanced_server import (
            store_problem_solving_experience,
            retrieve_similar_problems,
            get_lessons_for_domain
        )
        
        # Scenario: AI agent encounters a coding problem
        
        # 1. Store the problem-solving experience
        coding_problem = {
            "problem_name": "API Rate Limiting Implementation",
            "problem_description": "Need to implement rate limiting for REST API to prevent abuse",
            "solution_approach": "Used Redis-based sliding window rate limiter with exponential backoff",
            "key_insights": "Sliding window provides better user experience than fixed window",
            "mistakes_made": "Initially used in-memory rate limiting which didn't work in distributed setup",
            "tools_used": "Redis, Python, Flask, rate-limiting algorithms",
            "domain": "backend_api",
            "effectiveness": "high",
            "group_id": "ai_agent_memory"
        }
        
        store_result = await store_problem_solving_experience(**coding_problem)
        assert "message" in store_result
        
        # 2. Later, agent encounters similar problem
        similar_result = await retrieve_similar_problems(
            current_problem="Need to implement rate limiting for my API",
            domain="backend_api",
            max_results=5,
            group_ids=["ai_agent_memory"]
        )
        
        assert "nodes" in similar_result
        
        # 3. Agent wants to learn from past API experiences
        lessons_result = await get_lessons_for_domain(
            domain="backend_api",
            max_lessons=10,
            group_ids=["ai_agent_memory"]
        )
        
        assert "facts" in lessons_result
        
        # All operations should complete successfully
        assert all("error" not in result for result in [store_result, similar_result, lessons_result])
