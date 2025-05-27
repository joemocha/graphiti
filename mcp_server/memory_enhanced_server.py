#!/usr/bin/env python3
"""
Standalone Enhanced Graphiti MCP Server with AI Agent Memory capabilities.
Provides memory-specific tools and entity types for AI agents.
"""

import argparse
import asyncio
import logging
import os
import sys
from collections.abc import Callable
from datetime import datetime, timezone
from typing import Any, TypedDict, cast

from azure.identity import DefaultAzureCredential, get_bearer_token_provider
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP
from openai import AsyncAzureOpenAI
from pydantic import BaseModel, Field

from graphiti_core import Graphiti
from graphiti_core.cross_encoder.client import CrossEncoderClient
from graphiti_core.cross_encoder.openai_reranker_client import OpenAIRerankerClient
from graphiti_core.edges import EntityEdge
from graphiti_core.embedder.client import EmbedderClient
from graphiti_core.embedder.openai import OpenAIEmbedder, OpenAIEmbedderConfig
from graphiti_core.llm_client import LLMClient
from graphiti_core.llm_client.config import LLMConfig
from graphiti_core.llm_client.openai_client import OpenAIClient
from graphiti_core.nodes import EpisodeType
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

# Import memory-specific entity types
from memory_entity_types import MEMORY_ENTITY_TYPES

load_dotenv()

DEFAULT_LLM_MODEL = 'gpt-4.1-mini'
SMALL_LLM_MODEL = 'gpt-4.1-nano'
DEFAULT_EMBEDDER_MODEL = 'text-embedding-3-small'


# Base entity types from the original server
class Requirement(BaseModel):
    """A Requirement represents a specific need, feature, or functionality that a product or service must fulfill."""

    project_name: str = Field(
        ...,
        description='The name of the project to which the requirement belongs.',
    )
    description: str = Field(
        ...,
        description='Description of the requirement. Only use information mentioned in the context to write this description.',
    )


class Preference(BaseModel):
    """A Preference represents a user's expressed like, dislike, or preference for something."""

    category: str = Field(
        ...,
        description="The category of the preference. (e.g., 'Brands', 'Food', 'Music')",
    )
    description: str = Field(
        ...,
        description='Brief description of the preference. Only use information mentioned in the context to write this description.',
    )


class Procedure(BaseModel):
    """A Procedure informing the agent what actions to take or how to perform in certain scenarios."""

    description: str = Field(
        ...,
        description='Brief description of the procedure. Only use information mentioned in the context to write this description.',
    )


ENTITY_TYPES: dict[str, BaseModel] = {
    'Requirement': Requirement,  # type: ignore
    'Preference': Preference,  # type: ignore
    'Procedure': Procedure,  # type: ignore
}


# Type definitions for API responses
class ErrorResponse(TypedDict):
    error: str


class SuccessResponse(TypedDict):
    message: str


class NodeResult(TypedDict):
    uuid: str
    name: str
    summary: str
    labels: list[str]
    group_id: str
    created_at: str
    attributes: dict[str, Any]


class NodeSearchResponse(TypedDict):
    message: str
    nodes: list[NodeResult]


class FactSearchResponse(TypedDict):
    message: str
    facts: list[dict[str, Any]]


class EpisodeSearchResponse(TypedDict):
    message: str
    episodes: list[dict[str, Any]]


class StatusResponse(TypedDict):
    status: str
    message: str


def create_azure_credential_token_provider() -> Callable[[], str]:
    credential = DefaultAzureCredential()
    token_provider = get_bearer_token_provider(
        credential, 'https://cognitiveservices.azure.com/.default'
    )
    return token_provider


# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    stream=sys.stderr,
)
logger = logging.getLogger(__name__)

# Server configuration classes
class GraphitiLLMConfig(BaseModel):
    """Configuration for the LLM client."""

    api_key: str | None = None
    model: str = DEFAULT_LLM_MODEL
    small_model: str = SMALL_LLM_MODEL
    temperature: float = 0.0
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiLLMConfig':
        """Create LLM configuration from environment variables."""
        model_env = os.environ.get('MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_LLM_MODEL

        small_model_env = os.environ.get('SMALL_MODEL_NAME', '')
        small_model = small_model_env if small_model_env.strip() else SMALL_LLM_MODEL

        azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', None)
        azure_openai_api_version = os.environ.get('AZURE_OPENAI_API_VERSION', None)
        azure_openai_deployment_name = os.environ.get('AZURE_OPENAI_DEPLOYMENT_NAME', None)
        azure_openai_use_managed_identity = (
            os.environ.get('AZURE_OPENAI_USE_MANAGED_IDENTITY', 'false').lower() == 'true'
        )

        if azure_openai_endpoint is None:
            return cls(
                api_key=os.environ.get('OPENAI_API_KEY'),
                model=model,
                small_model=small_model,
                temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
            )
        else:
            if azure_openai_deployment_name is None:
                logger.error('AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set')
                raise ValueError('AZURE_OPENAI_DEPLOYMENT_NAME environment variable not set')

            if not azure_openai_use_managed_identity:
                api_key = os.environ.get('OPENAI_API_KEY', None)
            else:
                api_key = None

            return cls(
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                azure_openai_endpoint=azure_openai_endpoint,
                api_key=api_key,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
                model=model,
                small_model=small_model,
                temperature=float(os.environ.get('LLM_TEMPERATURE', '0.0')),
            )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiLLMConfig':
        """Create LLM configuration from CLI arguments, falling back to environment variables."""
        config = cls.from_env()

        if hasattr(args, 'model') and args.model:
            if args.model.strip():
                config.model = args.model
            else:
                logger.warning(f'Empty model name provided, using default: {DEFAULT_LLM_MODEL}')

        if hasattr(args, 'small_model') and args.small_model:
            if args.small_model.strip():
                config.small_model = args.small_model
            else:
                logger.warning(f'Empty small_model name provided, using default: {SMALL_LLM_MODEL}')

        if hasattr(args, 'temperature') and args.temperature is not None:
            config.temperature = args.temperature

        return config

    def create_client(self) -> LLMClient | None:
        """Create an LLM client based on this configuration."""
        if self.azure_openai_endpoint is not None:
            if self.azure_openai_use_managed_identity:
                token_provider = create_azure_credential_token_provider()
                return AsyncAzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    azure_deployment=self.azure_openai_deployment_name,
                    api_version=self.azure_openai_api_version,
                    azure_ad_token_provider=token_provider,
                )
            elif self.api_key:
                return AsyncAzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    azure_deployment=self.azure_openai_deployment_name,
                    api_version=self.azure_openai_api_version,
                    api_key=self.api_key,
                )
            else:
                logger.error('OPENAI_API_KEY must be set when using Azure OpenAI API')
                return None

        if not self.api_key:
            return None

        llm_client_config = LLMConfig(
            api_key=self.api_key, model=self.model, small_model=self.small_model
        )
        llm_client_config.temperature = self.temperature
        return OpenAIClient(config=llm_client_config)

    def create_cross_encoder_client(self) -> CrossEncoderClient | None:
        """Create a cross-encoder client based on this configuration."""
        if self.azure_openai_endpoint is not None:
            client = self.create_client()
            return OpenAIRerankerClient(client=client)
        else:
            llm_client_config = LLMConfig(
                api_key=self.api_key, model=self.model, small_model=self.small_model
            )
            return OpenAIRerankerClient(config=llm_client_config)


class GraphitiEmbedderConfig(BaseModel):
    """Configuration for the embedder client."""

    model: str = DEFAULT_EMBEDDER_MODEL
    api_key: str | None = None
    azure_openai_endpoint: str | None = None
    azure_openai_deployment_name: str | None = None
    azure_openai_api_version: str | None = None
    azure_openai_use_managed_identity: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiEmbedderConfig':
        """Create embedder configuration from environment variables."""
        model_env = os.environ.get('EMBEDDER_MODEL_NAME', '')
        model = model_env if model_env.strip() else DEFAULT_EMBEDDER_MODEL

        azure_openai_endpoint = os.environ.get('AZURE_OPENAI_ENDPOINT', None)
        azure_openai_api_version = os.environ.get('AZURE_OPENAI_EMBEDDING_API_VERSION', None)
        azure_openai_deployment_name = os.environ.get(
            'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', None
        )
        azure_openai_use_managed_identity = (
            os.environ.get('AZURE_OPENAI_USE_MANAGED_IDENTITY', 'false').lower() == 'true'
        )

        if azure_openai_endpoint is not None:
            azure_openai_deployment_name = os.environ.get(
                'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME', None
            )
            if azure_openai_deployment_name is None:
                logger.error('AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set')
                raise ValueError(
                    'AZURE_OPENAI_EMBEDDING_DEPLOYMENT_NAME environment variable not set'
                )

            if not azure_openai_use_managed_identity:
                api_key = os.environ.get('OPENAI_API_KEY', None)
            else:
                api_key = None

            return cls(
                azure_openai_use_managed_identity=azure_openai_use_managed_identity,
                azure_openai_endpoint=azure_openai_endpoint,
                api_key=api_key,
                azure_openai_api_version=azure_openai_api_version,
                azure_openai_deployment_name=azure_openai_deployment_name,
            )
        else:
            return cls(
                model=model,
                api_key=os.environ.get('OPENAI_API_KEY'),
            )

    def create_client(self) -> EmbedderClient | None:
        if self.azure_openai_endpoint is not None:
            if self.azure_openai_use_managed_identity:
                token_provider = create_azure_credential_token_provider()
                return AsyncAzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    azure_deployment=self.azure_openai_deployment_name,
                    api_version=self.azure_openai_api_version,
                    azure_ad_token_provider=token_provider,
                )
            elif self.api_key:
                return AsyncAzureOpenAI(
                    azure_endpoint=self.azure_openai_endpoint,
                    azure_deployment=self.azure_openai_deployment_name,
                    api_version=self.azure_openai_api_version,
                    api_key=self.api_key,
                )
            else:
                logger.error('OPENAI_API_KEY must be set when using Azure OpenAI API')
                return None
        else:
            if not self.api_key:
                return None

            embedder_config = OpenAIEmbedderConfig(api_key=self.api_key, embedding_model=self.model)
            return OpenAIEmbedder(config=embedder_config)


class Neo4jConfig(BaseModel):
    """Configuration for Neo4j database connection."""

    uri: str = 'bolt://localhost:7687'
    user: str = 'neo4j'
    password: str = 'password'

    @classmethod
    def from_env(cls) -> 'Neo4jConfig':
        """Create Neo4j configuration from environment variables."""
        return cls(
            uri=os.environ.get('NEO4J_URI', 'bolt://localhost:7687'),
            user=os.environ.get('NEO4J_USER', 'neo4j'),
            password=os.environ.get('NEO4J_PASSWORD', 'password'),
        )


class GraphitiConfig(BaseModel):
    """Configuration for Graphiti client."""

    llm: GraphitiLLMConfig = Field(default_factory=GraphitiLLMConfig)
    embedder: GraphitiEmbedderConfig = Field(default_factory=GraphitiEmbedderConfig)
    neo4j: Neo4jConfig = Field(default_factory=Neo4jConfig)
    group_id: str | None = None
    use_custom_entities: bool = False
    destroy_graph: bool = False

    @classmethod
    def from_env(cls) -> 'GraphitiConfig':
        """Create a configuration instance from environment variables."""
        return cls(
            llm=GraphitiLLMConfig.from_env(),
            embedder=GraphitiEmbedderConfig.from_env(),
            neo4j=Neo4jConfig.from_env(),
        )

    @classmethod
    def from_cli_and_env(cls, args: argparse.Namespace) -> 'GraphitiConfig':
        """Create configuration from CLI arguments, falling back to environment variables."""
        config = cls.from_env()

        if args.group_id:
            config.group_id = args.group_id
        else:
            config.group_id = 'default'

        config.use_custom_entities = args.use_custom_entities
        config.destroy_graph = args.destroy_graph
        config.llm = GraphitiLLMConfig.from_cli_and_env(args)

        return config


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    transport: str = 'sse'

    @classmethod
    def from_cli(cls, args: argparse.Namespace) -> 'MCPConfig':
        """Create MCP configuration from CLI arguments."""
        return cls(transport=args.transport)


# Create global config instance - will be properly initialized later
config = GraphitiConfig()

# Initialize Graphiti client
graphiti_client: Graphiti | None = None

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


async def initialize_memory_graphiti():
    """Initialize the Graphiti client with the configured settings."""
    global graphiti_client, config

    try:
        # Create LLM client if possible
        llm_client = config.llm.create_client()
        if not llm_client and config.use_custom_entities:
            # If custom entities are enabled, we must have an LLM client
            raise ValueError('OPENAI_API_KEY must be set when custom entities are enabled')

        # Log Neo4j configuration for debugging
        logger.info(f'Neo4j URI: {config.neo4j.uri}')
        logger.info(f'Neo4j User: {config.neo4j.user}')

        # Validate Neo4j configuration
        if not config.neo4j.uri or not config.neo4j.user or not config.neo4j.password:
            raise ValueError(
                'NEO4J_URI, NEO4J_USER, and NEO4J_PASSWORD must be set'
            )

        embedder_client = config.embedder.create_client()
        cross_encoder_client = config.llm.create_cross_encoder_client()

        # Initialize Graphiti client
        graphiti_client = Graphiti(
            uri=config.neo4j.uri,
            user=config.neo4j.user,
            password=config.neo4j.password,
            llm_client=llm_client,
            embedder=embedder_client,
            cross_encoder=cross_encoder_client,
        )

        # Destroy graph if requested
        if config.destroy_graph:
            logger.info('Destroying graph...')
            await clear_data(graphiti_client.driver)

        # Initialize the graph database with Graphiti's indices
        await graphiti_client.build_indices_and_constraints()
        logger.info('Graphiti client initialized successfully')

        # Log configuration details for transparency
        if llm_client:
            logger.info(f'Using OpenAI model: {config.llm.model}')
            logger.info(f'Using temperature: {config.llm.temperature}')
        else:
            logger.info('No LLM client configured - entity extraction will be limited')

        logger.info(f'Using group_id: {config.group_id}')
        logger.info(
            f'Custom entity extraction: {"enabled" if config.use_custom_entities else "disabled"}'
        )

    except Exception as e:
        logger.error(f'Failed to initialize Graphiti: {str(e)}')
        raise


def format_fact_result(edge: EntityEdge) -> dict[str, Any]:
    """Format an entity edge into a readable result."""
    return edge.model_dump(
        mode='json',
        exclude={
            'fact_embedding',
        },
    )


# Dictionary to store queues for each group_id
episode_queues: dict[str, asyncio.Queue] = {}
# Dictionary to track if a worker is running for each group_id
queue_workers: dict[str, bool] = {}


async def process_episode_queue(group_id: str):
    """Process episodes for a specific group_id sequentially."""
    global queue_workers

    logger.info(f'Starting episode queue worker for group_id: {group_id}')
    queue_workers[group_id] = True

    try:
        while True:
            # Get the next episode processing function from the queue
            process_func = await episode_queues[group_id].get()

            try:
                # Process the episode
                await process_func()
            except Exception as e:
                logger.error(f'Error processing queued episode for group_id {group_id}: {str(e)}')
            finally:
                # Mark the task as done regardless of success/failure
                episode_queues[group_id].task_done()
    except asyncio.CancelledError:
        logger.info(f'Episode queue worker for group_id {group_id} was cancelled')
    except Exception as e:
        logger.error(f'Unexpected error in queue worker for group_id {group_id}: {str(e)}')
    finally:
        queue_workers[group_id] = False
        logger.info(f'Stopped episode queue worker for group_id: {group_id}')


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


# Note: Base tools (add_memory, search_memory_nodes, etc.) are intentionally excluded
# from this memory-enhanced server to maintain separation of concerns.
# This server focuses only on memory-specific functionality.


@mcp.resource('http://graphiti/status')
async def get_status() -> StatusResponse:
    """Get the status of the Graphiti MCP server and Neo4j connection."""
    global graphiti_client

    if graphiti_client is None:
        return {'status': 'error', 'message': 'Graphiti client not initialized'}

    try:
        assert graphiti_client is not None
        client = cast(Graphiti, graphiti_client)

        await client.driver.verify_connectivity()
        return {'status': 'ok', 'message': 'Graphiti MCP server is running and connected to Neo4j'}
    except Exception as e:
        error_msg = str(e)
        logger.error(f'Error checking Neo4j connection: {error_msg}')
        return {
            'status': 'error',
            'message': f'Graphiti MCP server is running but Neo4j connection failed: {error_msg}',
        }


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

    # Load configuration from environment variables and CLI arguments
    new_config = GraphitiConfig.from_cli_and_env(args)

    # Debug logging to see what configuration was loaded
    logger.info(f'Loaded Neo4j URI from config: {new_config.neo4j.uri}')
    logger.info(f'Loaded Neo4j User from config: {new_config.neo4j.user}')
    logger.info(f'Environment NEO4J_URI: {os.environ.get("NEO4J_URI", "NOT SET")}')

    # Set default group_id for memory server if not provided
    if not new_config.group_id or new_config.group_id == 'default':
        new_config.group_id = 'memory_default'

    # Update the global config
    global config
    config = new_config

    # Initialize Graphiti with enhanced entity types
    try:
        await initialize_memory_graphiti()
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
