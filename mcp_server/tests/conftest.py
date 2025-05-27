"""
Pytest configuration and shared fixtures for MCP server tests.
"""

import asyncio
import os
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import AsyncGenerator, Dict, Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from dotenv import load_dotenv

# Add the mcp_server directory to the Python path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Load environment variables for testing
load_dotenv()


@pytest.fixture(scope="session")
def event_loop():
    """Create an instance of the default event loop for the test session."""
    loop = asyncio.get_event_loop_policy().new_event_loop()
    yield loop
    loop.close()


@pytest.fixture
def mock_graphiti_client():
    """Mock Graphiti client for testing."""
    client = AsyncMock()

    # Mock common methods
    client.add_episode = AsyncMock(return_value=None)
    client._search_nodes = AsyncMock(return_value=MagicMock(nodes=[]))
    client.search = AsyncMock(return_value=[])

    return client


@pytest.fixture
def mock_config():
    """Mock configuration for testing."""
    config = MagicMock()
    config.group_id = "test_group"
    config.use_custom_entities = True
    return config


@pytest.fixture
def sample_problem_data():
    """Sample problem-solving data for testing."""
    return {
        "problem_name": "API Performance Issue",
        "problem_description": "Database queries are taking too long, causing API timeouts",
        "solution_approach": "Added database indexing and implemented query optimization",
        "key_insights": "Most queries were missing proper indexes on foreign key columns",
        "mistakes_made": "Initially tried to optimize application code before checking database",
        "tools_used": "PostgreSQL EXPLAIN, pgAdmin, application profiler",
        "domain": "backend",
        "effectiveness": "high",
        "group_id": "test_memory"
    }


@pytest.fixture
def sample_entity_data():
    """Sample entity data for testing custom entity types."""
    return {
        "problem_solution": {
            "problem_type": "performance_optimization",
            "domain": "backend_api",
            "approach": "Database indexing and query optimization",
            "effectiveness": "high",
            "tools_used": "PostgreSQL, pgAdmin",
            "complexity": "moderate"
        },
        "lesson_learned": {
            "context": "API performance optimization project",
            "insight": "Always check database performance before optimizing application code",
            "applicability": "Any database-backed application with performance issues",
            "confidence_level": "high",
            "domain": "backend"
        },
        "common_mistake": {
            "mistake_type": "premature_optimization",
            "description": "Optimizing application code before identifying the actual bottleneck",
            "prevention": "Always profile and measure before optimizing",
            "warning_signs": "Assumptions about performance without data",
            "domain": "backend",
            "severity": "medium"
        }
    }


@pytest.fixture
def test_environment_vars():
    """Test environment variables."""
    return {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "test",
        "OPENAI_API_KEY": "test-key",
        "MODEL_NAME": "gpt-4o-mini"
    }


@pytest.fixture
def mock_server_setup(mock_graphiti_client, mock_config):
    """Setup mock server environment for testing."""
    # Use patch to mock the global variables
    with patch('memory_enhanced_server.graphiti_client', mock_graphiti_client), \
         patch('memory_enhanced_server.config', mock_config):
        yield {
            "client": mock_graphiti_client,
            "config": mock_config
        }
