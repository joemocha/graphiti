#!/usr/bin/env python3
"""
Enhanced Graphiti MCP Server with AI Agent Memory capabilities.
Extends the base MCP server with memory-specific tools and entity types.
"""

import asyncio
import logging
import sys
from datetime import datetime, timezone

from mcp.server.fastmcp import FastMCP
from graphiti_core.nodes import EpisodeType

# Import base server components
from graphiti_mcp_server import (
    graphiti_client,
    initialize_graphiti,
    config,
    ENTITY_TYPES,
    ErrorResponse,
    SuccessResponse,
    NodeSearchResponse,
    FactSearchResponse,
)

# Import memory-specific entity types
from memory_entity_types import MEMORY_ENTITY_TYPES

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Enhanced entity types combining base and memory types
ENHANCED_ENTITY_TYPES = {**ENTITY_TYPES, **MEMORY_ENTITY_TYPES}

# Enhanced MCP server instructions
MEMORY_MCP_INSTRUCTIONS = """
Enhanced Graphiti Memory Server for AI Agents - This server provides advanced memory capabilities
specifically designed for AI agents to store and retrieve problem-solving experiences.

Core Memory Capabilities:
1. Problem-Solution Storage: Store detailed problem definitions and their solutions
2. Lesson Learning: Capture insights and patterns from experiences
3. Mistake Prevention: Record common mistakes and prevention strategies
4. Context Awareness: Maintain environmental and situational context
5. Success Patterns: Identify and store repeatable successful approaches

Memory-Enhanced Tools:
- store_problem_solving_experience: Store a complete problem-solving session
- retrieve_similar_problems: Find problems similar to current challenge
- get_lessons_for_domain: Retrieve lessons learned for specific domain
- find_common_mistakes: Get common mistakes to avoid for problem type
- store_insight: Store a specific insight or lesson learned
- get_success_patterns: Retrieve proven successful patterns

Entity Types Available:
- ProblemSolution: Successful approaches to specific problems
- LessonLearned: Insights and patterns from experiences
- CommonMistake: Errors to avoid and prevention strategies
- ProblemContext: Environmental factors and constraints
- SuccessPattern: Repeatable approaches that work well

Use structured tagging and descriptive names for optimal memory organization and retrieval.
"""

# Create enhanced MCP server instance
mcp = FastMCP(
    'Enhanced Graphiti Agent Memory',
    instructions=MEMORY_MCP_INSTRUCTIONS,
    host="0.0.0.0",
    port=8000,  # Different port from base server
)


@mcp.tool()
async def store_problem_solving_experience(
    problem_name: str,
    problem_description: str,
    solution_approach: str,
    key_insights: str,
    mistakes_made: str = "",
    tools_used: str = "",
    domain: str = "",
    effectiveness: str = "medium",
    group_id: str | None = None,
) -> SuccessResponse | ErrorResponse:
    """Store a complete problem-solving experience with all components.

    This tool stores a comprehensive problem-solving session including the problem,
    solution, insights, and any mistakes made. It creates multiple related episodes
    that can be retrieved together or separately.

    Args:
        problem_name: Brief name/title for the problem
        problem_description: Detailed description of the problem and context
        solution_approach: The approach taken to solve the problem
        key_insights: Important insights or lessons learned
        mistakes_made: Any mistakes made during problem-solving (optional)
        tools_used: Tools, technologies, or frameworks used (optional)
        domain: Domain or field (e.g., 'backend', 'frontend', 'devops')
        effectiveness: How effective the solution was (low/medium/high)
        group_id: Group ID for organizing memories
    """
    global graphiti_client, config

    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        effective_group_id = group_id if group_id is not None else config.group_id
        if not effective_group_id:
            effective_group_id = 'memory_default'

        # Store problem definition episode
        problem_episode_body = f"""
        PROBLEM_TYPE: {domain}_problem
        DOMAIN: {domain}
        DESCRIPTION: {problem_description}
        CONTEXT: Problem-solving session
        TOOLS_AVAILABLE: {tools_used}
        TAGS: problem, {domain}, memory, problem_solving
        """

        await graphiti_client.add_episode(
            name=f"Problem: {problem_name}",
            episode_body=problem_episode_body,
            source=EpisodeType.text,
            source_description='problem_definition',
            group_id=effective_group_id,
            reference_time=datetime.now(timezone.utc),
            entity_types=ENHANCED_ENTITY_TYPES,
        )

        # Store solution episode
        solution_episode_body = f"""
        APPROACH: {solution_approach}
        PROBLEM_REFERENCE: {problem_name}
        TOOLS_USED: {tools_used}
        EFFECTIVENESS: {effectiveness}
        DOMAIN: {domain}
        TAGS: solution, {domain}, memory, {effectiveness}_effectiveness
        """

        await graphiti_client.add_episode(
            name=f"Solution: {problem_name}",
            episode_body=solution_episode_body,
            source=EpisodeType.text,
            source_description='solution_approach',
            group_id=effective_group_id,
            reference_time=datetime.now(timezone.utc),
            entity_types=ENHANCED_ENTITY_TYPES,
        )

        # Store insights episode
        if key_insights:
            insights_episode_body = f"""
            KEY_INSIGHTS: {key_insights}
            PROBLEM_REFERENCE: {problem_name}
            DOMAIN: {domain}
            APPLICABILITY: Similar {domain} problems
            TAGS: insights, lessons, {domain}, memory
            """

            await graphiti_client.add_episode(
                name=f"Insights: {problem_name}",
                episode_body=insights_episode_body,
                source=EpisodeType.text,
                source_description='insights_and_lessons',
                group_id=effective_group_id,
                reference_time=datetime.now(timezone.utc),
                entity_types=ENHANCED_ENTITY_TYPES,
            )

        # Store mistakes episode if any
        if mistakes_made:
            mistakes_episode_body = f"""
            MISTAKES_MADE: {mistakes_made}
            PROBLEM_REFERENCE: {problem_name}
            DOMAIN: {domain}
            PREVENTION_CONTEXT: {solution_approach}
            TAGS: mistakes, corrections, {domain}, memory
            """

            await graphiti_client.add_episode(
                name=f"Mistakes: {problem_name}",
                episode_body=mistakes_episode_body,
                source=EpisodeType.text,
                source_description='mistakes_and_corrections',
                group_id=effective_group_id,
                reference_time=datetime.now(timezone.utc),
                entity_types=ENHANCED_ENTITY_TYPES,
            )

        return {'message': f'Successfully stored problem-solving experience: {problem_name}'}

    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error storing problem-solving experience: {error_msg}')
        return {'error': f'Error storing experience: {error_msg}'}


@mcp.tool()
async def retrieve_similar_problems(
    current_problem: str,
    domain: str = "",
    max_results: int = 5,
    group_ids: list[str] | None = None,
) -> NodeSearchResponse | ErrorResponse:
    """Retrieve problems similar to the current challenge.

    Args:
        current_problem: Description of the current problem
        domain: Domain to focus search on (optional)
        max_results: Maximum number of similar problems to return
        group_ids: List of group IDs to search in
    """
    global graphiti_client, config

    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        # Construct search query
        search_query = f"problem {current_problem}"
        if domain:
            search_query += f" {domain}"

        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        # Search for similar problems using the search method
        from graphiti_core.search.search_config_recipes import NODE_HYBRID_SEARCH_RRF
        search_config = NODE_HYBRID_SEARCH_RRF.model_copy(deep=True)
        search_config.limit = max_results

        search_results = await graphiti_client._search(
            query=search_query,
            config=search_config,
            group_ids=effective_group_ids,
        )
        relevant_nodes = search_results

        if not relevant_nodes.nodes:
            return {'message': 'No similar problems found', 'nodes': []}

        # Format results
        formatted_nodes = []
        for node in relevant_nodes.nodes:
            formatted_nodes.append({
                'uuid': node.uuid,
                'name': node.name,
                'summary': node.summary,
                'labels': node.labels,
                'group_id': node.group_id,
                'created_at': node.created_at.isoformat(),
                'attributes': node.attributes,
            })

        return {'message': 'Similar problems retrieved successfully', 'nodes': formatted_nodes}

    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error retrieving similar problems: {error_msg}')
        return {'error': f'Error retrieving similar problems: {error_msg}'}


@mcp.tool()
async def get_lessons_for_domain(
    domain: str,
    max_lessons: int = 10,
    group_ids: list[str] | None = None,
) -> FactSearchResponse | ErrorResponse:
    """Retrieve lessons learned for a specific domain.

    Args:
        domain: The domain to get lessons for
        max_lessons: Maximum number of lessons to return
        group_ids: List of group IDs to search in
    """
    global graphiti_client, config

    if graphiti_client is None:
        return {'error': 'Graphiti client not initialized'}

    try:
        search_query = f"insights lessons {domain}"

        effective_group_ids = (
            group_ids if group_ids is not None else [config.group_id] if config.group_id else []
        )

        relevant_edges = await graphiti_client.search(
            group_ids=effective_group_ids,
            query=search_query,
            num_results=max_lessons,
        )

        if not relevant_edges:
            return {'message': f'No lessons found for domain: {domain}', 'facts': []}

        # Format facts
        facts = []
        for edge in relevant_edges:
            facts.append({
                'uuid': edge.uuid,
                'fact': edge.fact,
                'source_node_uuid': edge.source_node_uuid,
                'target_node_uuid': edge.target_node_uuid,
                'created_at': edge.created_at.isoformat(),
                'valid_at': edge.valid_at.isoformat() if edge.valid_at else None,
                'invalid_at': edge.invalid_at.isoformat() if edge.invalid_at else None,
            })

        return {'message': f'Lessons for {domain} retrieved successfully', 'facts': facts}

    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error retrieving lessons for domain: {error_msg}')
        return {'error': f'Error retrieving lessons: {error_msg}'}


# Import and expose base tools from the original server
from graphiti_mcp_server import (
    add_memory,
    search_memory_nodes,
    search_memory_facts,
    get_episodes,
    delete_episode,
    delete_entity_edge,
    get_entity_edge,
    clear_graph,
    get_status,
)

# Re-register base tools with the enhanced server
mcp.tool()(add_memory)
mcp.tool()(search_memory_nodes)
mcp.tool()(search_memory_facts)
mcp.tool()(get_episodes)
mcp.tool()(delete_episode)
mcp.tool()(delete_entity_edge)
mcp.tool()(get_entity_edge)
mcp.tool()(clear_graph)
mcp.tool()(get_status)


async def main():
    """Main function to run the enhanced MCP server."""
    import argparse
    import os
    import signal
    from dotenv import load_dotenv

    # Load environment variables first
    load_dotenv()

    parser = argparse.ArgumentParser(description='Enhanced Graphiti MCP Server with Memory')
    parser.add_argument('--group-id', help='Group ID for memory organization')
    parser.add_argument('--use-custom-entities', action='store_true',
                       help='Enable enhanced entity extraction')
    parser.add_argument('--transport', choices=['stdio', 'sse'], default='sse',
                       help='Transport method for MCP')
    parser.add_argument('--destroy-graph', action='store_true',
                       help='Destroy the graph on startup')

    args = parser.parse_args()

    # Use proper configuration loading from base server
    # Import and update the global config from the base module
    import graphiti_mcp_server
    from graphiti_mcp_server import GraphitiConfig

    # Load configuration from environment variables and CLI arguments
    new_config = GraphitiConfig.from_cli_and_env(args)

    # Debug logging to see what configuration was loaded
    logger.info(f'Loaded Neo4j URI from config: {new_config.neo4j.uri}')
    logger.info(f'Loaded Neo4j User from config: {new_config.neo4j.user}')
    logger.info(f'Environment NEO4J_URI: {os.environ.get("NEO4J_URI", "NOT SET")}')

    # Set default group_id for memory server if not provided
    if not new_config.group_id or new_config.group_id == 'default':
        new_config.group_id = 'memory_default'

    # Update the global config in both modules
    global config
    config = new_config
    graphiti_mcp_server.config = new_config

    # Initialize Graphiti with enhanced entity types
    try:
        await initialize_graphiti()
        logger.info('Graphiti initialization completed successfully')
    except Exception as e:
        logger.error(f'Failed to initialize Graphiti: {str(e)}')
        raise

    # Update entity types to include memory types
    if config.use_custom_entities:
        logger.info('Enhanced entity extraction enabled with memory types')

    # Set up graceful shutdown handling
    shutdown_event = asyncio.Event()

    def signal_handler(signum, frame):
        logger.info(f'Received signal {signum}, initiating graceful shutdown...')
        shutdown_event.set()

    signal.signal(signal.SIGTERM, signal_handler)
    signal.signal(signal.SIGINT, signal_handler)

    # Run the server with proper error handling
    try:
        logger.info(f'Starting MCP server with transport: {args.transport}')
        if args.transport == 'stdio':
            await mcp.run_stdio()
        else:
            # Use the async version with proper error handling
            server_task = asyncio.create_task(mcp.run_sse_async())
            shutdown_task = asyncio.create_task(shutdown_event.wait())

            # Wait for either server completion or shutdown signal
            done, pending = await asyncio.wait(
                [server_task, shutdown_task],
                return_when=asyncio.FIRST_COMPLETED
            )

            # Cancel any pending tasks
            for task in pending:
                task.cancel()
                try:
                    await task
                except asyncio.CancelledError:
                    pass

            # Check if server task completed with an exception
            if server_task in done:
                try:
                    await server_task
                except Exception as e:
                    logger.error(f'Server task failed: {str(e)}')
                    raise

    except KeyboardInterrupt:
        logger.info('Received keyboard interrupt, shutting down...')
    except Exception as e:
        logger.error(f'Server error: {str(e)}')
        raise
    finally:
        logger.info('Server shutdown complete')


if __name__ == '__main__':
    asyncio.run(main())
