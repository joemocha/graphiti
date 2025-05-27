"""
Unit tests for memory enhanced server functions.
Tests the memory-specific tools and their functionality.
"""

import pytest
from datetime import datetime, timezone
from unittest.mock import AsyncMock, MagicMock, patch

from memory_enhanced_server import (
    store_problem_solving_experience,
    retrieve_similar_problems,
    get_lessons_for_domain,
    ENHANCED_ENTITY_TYPES,
)


class TestStoreProblemSolvingExperience:
    """Test the store_problem_solving_experience function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_complete_experience(self, mock_server_setup, sample_problem_data):
        """Test storing a complete problem-solving experience."""
        mock_client = mock_server_setup["client"]
        
        result = await store_problem_solving_experience(**sample_problem_data)
        
        # Verify success response
        assert "message" in result
        assert "Successfully stored problem-solving experience" in result["message"]
        assert sample_problem_data["problem_name"] in result["message"]
        
        # Verify that add_episode was called multiple times (problem, solution, insights, mistakes)
        assert mock_client.add_episode.call_count == 4
        
        # Verify the calls were made with correct parameters
        calls = mock_client.add_episode.call_args_list
        
        # Check problem episode
        problem_call = calls[0]
        assert f"Problem: {sample_problem_data['problem_name']}" in problem_call[1]["name"]
        assert sample_problem_data["domain"] in problem_call[1]["episode_body"]
        assert sample_problem_data["problem_description"] in problem_call[1]["episode_body"]
        
        # Check solution episode
        solution_call = calls[1]
        assert f"Solution: {sample_problem_data['problem_name']}" in solution_call[1]["name"]
        assert sample_problem_data["solution_approach"] in solution_call[1]["episode_body"]
        assert sample_problem_data["effectiveness"] in solution_call[1]["episode_body"]
        
        # Check insights episode
        insights_call = calls[2]
        assert f"Insights: {sample_problem_data['problem_name']}" in insights_call[1]["name"]
        assert sample_problem_data["key_insights"] in insights_call[1]["episode_body"]
        
        # Check mistakes episode
        mistakes_call = calls[3]
        assert f"Mistakes: {sample_problem_data['problem_name']}" in mistakes_call[1]["name"]
        assert sample_problem_data["mistakes_made"] in mistakes_call[1]["episode_body"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_minimal_experience(self, mock_server_setup):
        """Test storing experience with only required fields."""
        mock_client = mock_server_setup["client"]
        
        minimal_data = {
            "problem_name": "Simple Issue",
            "problem_description": "A simple problem",
            "solution_approach": "A simple solution",
            "key_insights": "Simple insight"
        }
        
        result = await store_problem_solving_experience(**minimal_data)
        
        # Verify success response
        assert "message" in result
        assert "Successfully stored problem-solving experience" in result["message"]
        
        # Should only call add_episode 3 times (problem, solution, insights - no mistakes)
        assert mock_client.add_episode.call_count == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_experience_no_insights(self, mock_server_setup):
        """Test storing experience without insights."""
        mock_client = mock_server_setup["client"]
        
        data = {
            "problem_name": "Issue Without Insights",
            "problem_description": "A problem",
            "solution_approach": "A solution",
            "key_insights": "",  # Empty insights
            "mistakes_made": "Some mistake"
        }
        
        result = await store_problem_solving_experience(**data)
        
        # Verify success response
        assert "message" in result
        
        # Should call add_episode 3 times (problem, solution, mistakes - no insights)
        assert mock_client.add_episode.call_count == 3

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_experience_client_not_initialized(self):
        """Test error handling when Graphiti client is not initialized."""
        # Mock the global graphiti_client as None
        with patch('memory_enhanced_server.graphiti_client', None):
            result = await store_problem_solving_experience(
                problem_name="test",
                problem_description="test",
                solution_approach="test",
                key_insights="test"
            )
        
        assert "error" in result
        assert "Graphiti client not initialized" in result["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_experience_exception_handling(self, mock_server_setup):
        """Test exception handling during storage."""
        mock_client = mock_server_setup["client"]
        mock_client.add_episode.side_effect = Exception("Database error")
        
        result = await store_problem_solving_experience(
            problem_name="test",
            problem_description="test", 
            solution_approach="test",
            key_insights="test"
        )
        
        assert "error" in result
        assert "Error storing experience" in result["error"]
        assert "Database error" in result["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_store_experience_group_id_handling(self, mock_server_setup, sample_problem_data):
        """Test group_id handling with different scenarios."""
        mock_client = mock_server_setup["client"]
        mock_config = mock_server_setup["config"]
        
        # Test with explicit group_id
        data_with_group = {**sample_problem_data, "group_id": "custom_group"}
        await store_problem_solving_experience(**data_with_group)
        
        # Verify custom group_id was used
        call_args = mock_client.add_episode.call_args_list[0]
        assert call_args[1]["group_id"] == "custom_group"
        
        # Reset mock
        mock_client.reset_mock()
        
        # Test with config group_id
        data_without_group = {k: v for k, v in sample_problem_data.items() if k != "group_id"}
        await store_problem_solving_experience(**data_without_group)
        
        # Verify config group_id was used
        call_args = mock_client.add_episode.call_args_list[0]
        assert call_args[1]["group_id"] == mock_config.group_id


class TestRetrieveSimilarProblems:
    """Test the retrieve_similar_problems function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_similar_problems_success(self, mock_server_setup):
        """Test successful retrieval of similar problems."""
        mock_client = mock_server_setup["client"]
        
        # Mock search results
        mock_node = MagicMock()
        mock_node.uuid = "test-uuid"
        mock_node.name = "Similar Problem"
        mock_node.summary = "A similar problem summary"
        mock_node.labels = ["Problem"]
        mock_node.group_id = "test_group"
        mock_node.created_at = datetime.now(timezone.utc)
        mock_node.attributes = {"domain": "backend"}
        
        mock_search_result = MagicMock()
        mock_search_result.nodes = [mock_node]
        mock_client._search_nodes.return_value = mock_search_result
        
        result = await retrieve_similar_problems(
            current_problem="API performance issue",
            domain="backend",
            max_results=5
        )
        
        # Verify success response
        assert "message" in result
        assert "Similar problems retrieved successfully" in result["message"]
        assert "nodes" in result
        assert len(result["nodes"]) == 1
        
        # Verify node data
        node_data = result["nodes"][0]
        assert node_data["uuid"] == "test-uuid"
        assert node_data["name"] == "Similar Problem"
        assert node_data["summary"] == "A similar problem summary"
        
        # Verify search was called correctly
        mock_client._search_nodes.assert_called_once()
        call_args = mock_client._search_nodes.call_args
        assert "API performance issue" in call_args[1]["query"]
        assert "backend" in call_args[1]["query"]
        assert call_args[1]["num_results"] == 5

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_similar_problems_no_results(self, mock_server_setup):
        """Test retrieval when no similar problems are found."""
        mock_client = mock_server_setup["client"]
        
        # Mock empty search results
        mock_search_result = MagicMock()
        mock_search_result.nodes = []
        mock_client._search_nodes.return_value = mock_search_result
        
        result = await retrieve_similar_problems(
            current_problem="Unique problem",
            domain="frontend"
        )
        
        # Verify response for no results
        assert "message" in result
        assert "No similar problems found" in result["message"]
        assert "nodes" in result
        assert len(result["nodes"]) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_similar_problems_client_not_initialized(self):
        """Test error handling when Graphiti client is not initialized."""
        with patch('memory_enhanced_server.graphiti_client', None):
            result = await retrieve_similar_problems(
                current_problem="test problem"
            )
        
        assert "error" in result
        assert "Graphiti client not initialized" in result["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_retrieve_similar_problems_exception_handling(self, mock_server_setup):
        """Test exception handling during retrieval."""
        mock_client = mock_server_setup["client"]
        mock_client._search_nodes.side_effect = Exception("Search error")
        
        result = await retrieve_similar_problems(
            current_problem="test problem"
        )
        
        assert "error" in result
        assert "Error retrieving similar problems" in result["error"]
        assert "Search error" in result["error"]


class TestGetLessonsForDomain:
    """Test the get_lessons_for_domain function."""

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_lessons_success(self, mock_server_setup):
        """Test successful retrieval of lessons for a domain."""
        mock_client = mock_server_setup["client"]
        
        # Mock search results
        mock_edge = MagicMock()
        mock_edge.uuid = "lesson-uuid"
        mock_edge.fact = "Always check database performance first"
        mock_edge.source_node_uuid = "source-uuid"
        mock_edge.target_node_uuid = "target-uuid"
        mock_edge.created_at = datetime.now(timezone.utc)
        mock_edge.valid_at = datetime.now(timezone.utc)
        mock_edge.invalid_at = None
        
        mock_client.search.return_value = [mock_edge]
        
        result = await get_lessons_for_domain(
            domain="backend",
            max_lessons=10
        )
        
        # Verify success response
        assert "message" in result
        assert "Lessons for backend retrieved successfully" in result["message"]
        assert "facts" in result
        assert len(result["facts"]) == 1
        
        # Verify fact data
        fact_data = result["facts"][0]
        assert fact_data["uuid"] == "lesson-uuid"
        assert fact_data["fact"] == "Always check database performance first"
        assert fact_data["source_node_uuid"] == "source-uuid"
        assert fact_data["target_node_uuid"] == "target-uuid"
        
        # Verify search was called correctly
        mock_client.search.assert_called_once()
        call_args = mock_client.search.call_args
        assert "insights lessons backend" in call_args[1]["query"]
        assert call_args[1]["num_results"] == 10

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_lessons_no_results(self, mock_server_setup):
        """Test retrieval when no lessons are found."""
        mock_client = mock_server_setup["client"]
        mock_client.search.return_value = []
        
        result = await get_lessons_for_domain(domain="unknown_domain")
        
        # Verify response for no results
        assert "message" in result
        assert "No lessons found for domain: unknown_domain" in result["message"]
        assert "facts" in result
        assert len(result["facts"]) == 0

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_lessons_client_not_initialized(self):
        """Test error handling when Graphiti client is not initialized."""
        with patch('memory_enhanced_server.graphiti_client', None):
            result = await get_lessons_for_domain(domain="test")
        
        assert "error" in result
        assert "Graphiti client not initialized" in result["error"]

    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_get_lessons_exception_handling(self, mock_server_setup):
        """Test exception handling during lesson retrieval."""
        mock_client = mock_server_setup["client"]
        mock_client.search.side_effect = Exception("Search error")
        
        result = await get_lessons_for_domain(domain="test")
        
        assert "error" in result
        assert "Error retrieving lessons" in result["error"]
        assert "Search error" in result["error"]


class TestEnhancedEntityTypes:
    """Test the ENHANCED_ENTITY_TYPES configuration."""

    @pytest.mark.unit
    def test_enhanced_entity_types_includes_memory_types(self):
        """Test that ENHANCED_ENTITY_TYPES includes memory entity types."""
        from memory_entity_types import MEMORY_ENTITY_TYPES
        
        # Verify memory types are included
        for name, entity_type in MEMORY_ENTITY_TYPES.items():
            assert name in ENHANCED_ENTITY_TYPES
            assert ENHANCED_ENTITY_TYPES[name] == entity_type

    @pytest.mark.unit
    def test_enhanced_entity_types_includes_base_types(self):
        """Test that ENHANCED_ENTITY_TYPES includes base entity types."""
        from graphiti_mcp_server import ENTITY_TYPES
        
        # Verify base types are included
        for name, entity_type in ENTITY_TYPES.items():
            assert name in ENHANCED_ENTITY_TYPES
            assert ENHANCED_ENTITY_TYPES[name] == entity_type
