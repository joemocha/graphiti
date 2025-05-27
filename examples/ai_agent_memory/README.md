# AI Agent Memory System Implementation

This directory contains a complete implementation of the AI Agent Memory System using Graphiti MCP, based on the patterns outlined in `PLAN.md`.

## Overview

The implementation provides:

1. **Enhanced MCP Server** (`memory_enhanced_server.py`) - Extended Graphiti MCP server with memory-specific tools and entity types
2. **Memory Client** (`memory_client.py`) - Client-side interface for memory-driven problem solving
3. **LangGraph Integration** (`langgraph_memory_agent.py`) - Example of integrating memory with LangGraph agents

## Architecture

```
┌─────────────────┐    ┌──────────────────┐    ┌─────────────────┐
│   AI Agent      │    │  Memory Client   │    │ Enhanced MCP    │
│   (LangGraph)   │◄──►│                  │◄──►│ Server          │
│                 │    │                  │    │                 │
└─────────────────┘    └──────────────────┘    └─────────────────┘
                                                         │
                                                         ▼
                                               ┌─────────────────┐
                                               │   Graphiti      │
                                               │   Knowledge     │
                                               │   Graph         │
                                               └─────────────────┘
```

## Components

### 1. Server-side Implementation

#### Enhanced Entity Types (`memory_entity_types.py`)
- **ProblemSolution**: Successful approaches to specific problems
- **LessonLearned**: Insights and patterns from experiences
- **CommonMistake**: Errors to avoid and prevention strategies
- **ProblemContext**: Environmental factors and constraints
- **SuccessPattern**: Repeatable approaches that work well

#### Enhanced MCP Server (`memory_enhanced_server.py`)
Additional tools beyond the base Graphiti MCP server:
- `store_problem_solving_experience`: Store complete problem-solving sessions
- `retrieve_similar_problems`: Find problems similar to current challenge
- `get_lessons_for_domain`: Retrieve lessons learned for specific domain
- `find_common_mistakes`: Get common mistakes to avoid for problem type

### 2. Client-side Implementation

#### Memory Client (`memory_client.py`)
Implements the memory-driven workflow:
- **Pre-task Memory Search**: Search for relevant past experiences
- **Memory Integration**: Integrate memories into problem-solving approach
- **Post-task Storage**: Store complete problem-solving sessions
- **Workflow Orchestration**: Complete memory-driven problem-solving workflow

#### LangGraph Integration (`langgraph_memory_agent.py`)
Shows how to integrate memory capabilities with LangGraph agents:
- Memory-enhanced agent state
- Automatic memory search before tasks
- Memory-informed system messages
- Tool integration for memory operations

## Setup Instructions

### 1. Prerequisites

```bash
# Install dependencies
pip install graphiti-core langchain-openai langgraph python-dotenv

# Ensure Neo4j is running
# Set environment variables:
export NEO4J_URI="bolt://localhost:7687"
export NEO4J_USER="neo4j"
export NEO4J_PASSWORD="your_password"
export OPENAI_API_KEY="your_openai_key"
```

### 2. Start Enhanced MCP Server

```bash
# From the mcp_server directory
python memory_enhanced_server.py --use-custom-entities --group-id memory_agent
```

### 3. Configure MCP Client

For Cursor IDE integration, create a configuration file:

```json
{
  "mcpServers": {
    "graphiti-memory": {
      "transport": "stdio",
      "command": "python",
      "args": [
        "/path/to/graphiti/mcp_server/memory_enhanced_server.py",
        "--transport", "stdio",
        "--use-custom-entities",
        "--group-id", "cursor_memory"
      ],
      "env": {
        "NEO4J_URI": "bolt://localhost:7687",
        "NEO4J_USER": "neo4j",
        "NEO4J_PASSWORD": "your_password",
        "OPENAI_API_KEY": "${OPENAI_API_KEY}"
      }
    }
  }
}
```

### 4. Run Examples

```bash
# Test the memory client
python memory_client.py

# Run the LangGraph memory agent
python langgraph_memory_agent.py
```

## Usage Patterns

### 1. Storing Problem-Solving Experiences

```python
# Store a complete debugging session
await memory_client.store_problem_solving_session(
    problem_name="API Timeout Issue",
    problem_description="API endpoints returning 504 timeout errors during peak traffic",
    solution_approach="Added connection pool monitoring and proper cleanup",
    key_insights="Always check connection pool metrics first for timeout issues",
    mistakes_made="Initially focused on server resources instead of connection pool",
    tools_used="postgresql_logs, application_monitoring, connection_profiler",
    domain="backend_api",
    effectiveness="high"
)
```

### 2. Memory-Driven Problem Solving

```python
# Search memory before starting a new task
memory_results = await memory_client.pre_task_memory_search(
    task_description="Database performance issues",
    task_domain="backend_database",
    task_type="performance_optimization"
)

# Integrate memories into approach
integration_strategy = await memory_client.integrate_memory_into_approach(
    task_description="Database performance issues",
    memory_results=memory_results,
    initial_approach="Check query performance and indexes"
)
```

### 3. LangGraph Agent with Memory

```python
# Initialize memory-enhanced agent
agent = MemoryEnhancedAgent(graphiti_client, agent_id="my_agent")

# Run task with automatic memory integration
result = await agent.run_task(
    task_description="Debug API performance issues",
    domain="backend_development",
    task_type="performance_debugging",
    user_message="Our API is slow during peak hours"
)
```

## Memory Organization

### Group IDs
Use group IDs to organize memories by:
- Agent instance (`agent_001`, `agent_002`)
- Project (`project_alpha`, `project_beta`)
- Domain (`backend_memories`, `frontend_memories`)
- Time period (`2024_q1`, `2024_q2`)

### Tagging Strategy
Consistent tagging improves retrieval:
- **Primary Category**: `problem`, `solution`, `insight`, `mistake`
- **Domain**: `backend`, `frontend`, `database`, `devops`
- **Complexity**: `simple`, `moderate`, `complex`, `expert`
- **Outcome**: `successful`, `partial`, `failed`

## Best Practices

1. **Consistent Storage**: Always store complete problem-solving sessions
2. **Rich Context**: Include environmental factors and constraints
3. **Specific Insights**: Store actionable, specific lessons learned
4. **Mistake Documentation**: Record failures and prevention strategies
5. **Regular Retrieval**: Search memory before starting new tasks
6. **Iterative Improvement**: Refine tagging and storage based on retrieval effectiveness

## Troubleshooting

### Common Issues

1. **MCP Server Connection**: Ensure server is running and accessible
2. **Neo4j Connection**: Verify Neo4j is running and credentials are correct
3. **Memory Retrieval**: Check group IDs and search queries
4. **Entity Extraction**: Ensure `--use-custom-entities` flag is set

### Debugging

```bash
# Check server status
curl http://localhost:8001/status

# View server logs
tail -f mcp_server.log

# Test memory storage
python -c "
import asyncio
from memory_client import *
# Test code here
"
```

## Next Steps

1. **Extend Entity Types**: Add domain-specific entity types
2. **Custom Tools**: Implement additional memory tools for specific use cases
3. **Integration**: Integrate with your existing agent framework
4. **Monitoring**: Add metrics and monitoring for memory effectiveness
5. **Optimization**: Tune search parameters and storage strategies

## Contributing

When extending this implementation:
1. Follow the patterns established in `PLAN.md`
2. Add comprehensive documentation
3. Include examples and tests
4. Maintain backward compatibility
5. Update this README with new features
