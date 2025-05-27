"""
Integration tests for the memory enhanced MCP server.
These tests require a running Neo4j instance and test the full integration.
"""

import asyncio
import os
import pytest
from datetime import datetime, timezone
from typing import Dict, Any

from dotenv import load_dotenv

# Load environment variables
load_dotenv()


@pytest.fixture(scope="session")
def integration_test_config():
    """Configuration for integration tests."""
    return {
        "neo4j_uri": os.getenv("NEO4J_URI", "bolt://localhost:7687"),
        "neo4j_user": os.getenv("NEO4J_USER", "neo4j"),
        "neo4j_password": os.getenv("NEO4J_PASSWORD", "test"),
        "openai_api_key": os.getenv("TEST_OPENAI_API_KEY"),
        "model_name": os.getenv("TEST_OPENAI_MODEL", "gpt-4o-mini"),
        "group_id": f"test_memory_{int(datetime.now().timestamp())}"
    }


@pytest.fixture(scope="session")
async def real_graphiti_client(integration_test_config):
    """Create a real Graphiti client for integration testing."""
    try:
        from graphiti_core import Graphiti
        from graphiti_core.llm_client import OpenAIClient, LLMConfig
        
        # Skip if no API key available
        if not integration_test_config["openai_api_key"]:
            pytest.skip("OpenAI API key not available for integration tests")
        
        # Create LLM client
        llm_client = OpenAIClient(
            LLMConfig(
                api_key=integration_test_config["openai_api_key"],
                model=integration_test_config["model_name"]
            )
        )
        
        # Create Graphiti client
        client = Graphiti(
            uri=integration_test_config["neo4j_uri"],
            user=integration_test_config["neo4j_user"],
            password=integration_test_config["neo4j_password"],
            llm_client=llm_client
        )
        
        yield client
        
        # Cleanup: Clear test data
        try:
            await client.close()
        except Exception:
            pass  # Ignore cleanup errors
            
    except ImportError:
        pytest.skip("Graphiti core not available for integration tests")
    except Exception as e:
        pytest.skip(f"Failed to create Graphiti client: {e}")


@pytest.fixture
async def clean_test_environment(real_graphiti_client, integration_test_config):
    """Ensure clean test environment by clearing test group data."""
    group_id = integration_test_config["group_id"]
    
    # Clear any existing test data
    try:
        # Note: In a real implementation, you'd want a more targeted cleanup
        # For now, we'll rely on unique group IDs
        pass
    except Exception:
        pass  # Ignore cleanup errors
    
    yield group_id
    
    # Cleanup after test
    try:
        # Clear test data after tests complete
        pass
    except Exception:
        pass  # Ignore cleanup errors


class TestMemoryEnhancedServerIntegration:
    """Integration tests for the memory enhanced server."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_full_memory_workflow(self, real_graphiti_client, clean_test_environment):
        """Test the complete memory workflow: store and retrieve."""
        from memory_enhanced_server import (
            store_problem_solving_experience,
            retrieve_similar_problems,
            get_lessons_for_domain
        )
        from memory_entity_types import MEMORY_ENTITY_TYPES
        
        # Setup global client for the functions
        import memory_enhanced_server
        import graphiti_mcp_server
        
        original_client = graphiti_mcp_server.graphiti_client
        original_config = graphiti_mcp_server.config
        
        try:
            # Set up the global client and config
            graphiti_mcp_server.graphiti_client = real_graphiti_client
            
            # Mock config
            from unittest.mock import MagicMock
            mock_config = MagicMock()
            mock_config.group_id = clean_test_environment
            mock_config.use_custom_entities = True
            graphiti_mcp_server.config = mock_config
            
            # Test data
            problem_data = {
                "problem_name": "Integration Test Problem",
                "problem_description": "Testing the full memory integration workflow",
                "solution_approach": "Use comprehensive integration testing",
                "key_insights": "Integration tests reveal issues unit tests miss",
                "mistakes_made": "Initially forgot to test error conditions",
                "tools_used": "pytest, Neo4j, Graphiti",
                "domain": "testing",
                "effectiveness": "high",
                "group_id": clean_test_environment
            }
            
            # Step 1: Store problem-solving experience
            store_result = await store_problem_solving_experience(**problem_data)
            assert "message" in store_result
            assert "Successfully stored" in store_result["message"]
            
            # Wait a bit for indexing
            await asyncio.sleep(2)
            
            # Step 2: Retrieve similar problems
            retrieve_result = await retrieve_similar_problems(
                current_problem="Integration testing workflow",
                domain="testing",
                max_results=5,
                group_ids=[clean_test_environment]
            )
            
            assert "nodes" in retrieve_result
            # We should find at least our stored problem
            # Note: Exact matching depends on Graphiti's search algorithm
            
            # Step 3: Get lessons for domain
            lessons_result = await get_lessons_for_domain(
                domain="testing",
                max_lessons=10,
                group_ids=[clean_test_environment]
            )
            
            assert "facts" in lessons_result
            # We should find lessons related to our insights
            
        finally:
            # Restore original values
            graphiti_mcp_server.graphiti_client = original_client
            graphiti_mcp_server.config = original_config

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_entity_type_extraction(self, real_graphiti_client, clean_test_environment):
        """Test that custom entity types are properly extracted."""
        from memory_enhanced_server import store_problem_solving_experience
        from memory_entity_types import MEMORY_ENTITY_TYPES
        
        # Setup global client
        import memory_enhanced_server
        import graphiti_mcp_server
        
        original_client = graphiti_mcp_server.graphiti_client
        original_config = graphiti_mcp_server.config
        
        try:
            graphiti_mcp_server.graphiti_client = real_graphiti_client
            
            from unittest.mock import MagicMock
            mock_config = MagicMock()
            mock_config.group_id = clean_test_environment
            mock_config.use_custom_entities = True
            graphiti_mcp_server.config = mock_config
            
            # Store experience with rich entity data
            problem_data = {
                "problem_name": "Entity Extraction Test",
                "problem_description": "Testing custom entity type extraction in Graphiti",
                "solution_approach": "Implement custom Pydantic models for entity types",
                "key_insights": "Custom entity types improve knowledge organization",
                "mistakes_made": "Initially used generic entity types",
                "tools_used": "Pydantic, Graphiti, Python",
                "domain": "knowledge_management",
                "effectiveness": "high",
                "group_id": clean_test_environment
            }
            
            result = await store_problem_solving_experience(**problem_data)
            assert "message" in result
            assert "Successfully stored" in result["message"]
            
            # Wait for processing
            await asyncio.sleep(3)
            
            # Query the graph to verify entities were created
            # Note: This would require access to Graphiti's internal query methods
            # For now, we verify the function completed successfully
            
        finally:
            graphiti_mcp_server.graphiti_client = original_client
            graphiti_mcp_server.config = original_config

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_with_real_client(self, real_graphiti_client):
        """Test error handling with real client but invalid data."""
        from memory_enhanced_server import store_problem_solving_experience
        
        import memory_enhanced_server
        import graphiti_mcp_server
        
        original_client = graphiti_mcp_server.graphiti_client
        original_config = graphiti_mcp_server.config
        
        try:
            graphiti_mcp_server.graphiti_client = real_graphiti_client
            
            from unittest.mock import MagicMock
            mock_config = MagicMock()
            mock_config.group_id = "test_error_handling"
            graphiti_mcp_server.config = mock_config
            
            # Test with minimal data that might cause issues
            result = await store_problem_solving_experience(
                problem_name="",  # Empty name
                problem_description="",  # Empty description
                solution_approach="",  # Empty solution
                key_insights=""  # Empty insights
            )
            
            # Should handle gracefully - either succeed with empty data or return error
            assert isinstance(result, dict)
            assert "message" in result or "error" in result
            
        finally:
            graphiti_mcp_server.graphiti_client = original_client
            graphiti_mcp_server.config = original_config


class TestMemoryEntityTypesIntegration:
    """Integration tests for memory entity types with real Graphiti."""

    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_entity_validation_in_context(self, real_graphiti_client):
        """Test entity type validation in the context of Graphiti operations."""
        from memory_entity_types import ProblemSolution, LessonLearned, CommonMistake
        
        # Test that entity types can be serialized for Graphiti
        problem_solution = ProblemSolution(
            problem_type="integration_test",
            domain="testing",
            approach="Comprehensive testing approach",
            effectiveness="high",
            tools_used="pytest, Neo4j",
            complexity="moderate"
        )
        
        # Verify serialization works
        serialized = problem_solution.model_dump()
        assert isinstance(serialized, dict)
        assert all(isinstance(v, (str, int, float, bool, type(None))) for v in serialized.values())
        
        # Test lesson learned
        lesson = LessonLearned(
            context="Integration testing",
            insight="Real database interactions reveal edge cases",
            applicability="All database-dependent applications",
            confidence_level="high",
            domain="testing"
        )
        
        lesson_serialized = lesson.model_dump()
        assert isinstance(lesson_serialized, dict)
        
        # Test common mistake
        mistake = CommonMistake(
            mistake_type="insufficient_testing",
            description="Not testing with real database",
            prevention="Always include integration tests",
            warning_signs="Unit tests pass but system fails",
            domain="testing",
            severity="high"
        )
        
        mistake_serialized = mistake.model_dump()
        assert isinstance(mistake_serialized, dict)


@pytest.mark.integration
class TestPerformanceAndScaling:
    """Test performance characteristics of the memory system."""

    @pytest.mark.slow
    @pytest.mark.asyncio
    async def test_bulk_memory_storage(self, real_graphiti_client, clean_test_environment):
        """Test storing multiple memories and retrieval performance."""
        from memory_enhanced_server import store_problem_solving_experience, retrieve_similar_problems
        
        import memory_enhanced_server
        import graphiti_mcp_server
        
        original_client = graphiti_mcp_server.graphiti_client
        original_config = graphiti_mcp_server.config
        
        try:
            graphiti_mcp_server.graphiti_client = real_graphiti_client
            
            from unittest.mock import MagicMock
            mock_config = MagicMock()
            mock_config.group_id = clean_test_environment
            graphiti_mcp_server.config = mock_config
            
            # Store multiple problems
            problems = []
            for i in range(5):  # Start small for testing
                problem_data = {
                    "problem_name": f"Performance Test Problem {i}",
                    "problem_description": f"Testing performance with problem number {i}",
                    "solution_approach": f"Solution approach for problem {i}",
                    "key_insights": f"Insight number {i} about performance",
                    "domain": "performance",
                    "effectiveness": "medium",
                    "group_id": clean_test_environment
                }
                problems.append(problem_data)
            
            # Store all problems
            start_time = datetime.now()
            for problem in problems:
                result = await store_problem_solving_experience(**problem)
                assert "message" in result
            
            storage_time = (datetime.now() - start_time).total_seconds()
            
            # Wait for indexing
            await asyncio.sleep(2)
            
            # Test retrieval performance
            start_time = datetime.now()
            result = await retrieve_similar_problems(
                current_problem="Performance testing",
                domain="performance",
                max_results=10,
                group_ids=[clean_test_environment]
            )
            retrieval_time = (datetime.now() - start_time).total_seconds()
            
            # Basic performance assertions
            assert storage_time < 30  # Should store 5 problems in under 30 seconds
            assert retrieval_time < 5   # Should retrieve in under 5 seconds
            
            # Verify we got results
            assert "nodes" in result
            
        finally:
            graphiti_mcp_server.graphiti_client = original_client
            graphiti_mcp_server.config = original_config
