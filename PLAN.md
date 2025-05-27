# Detailed Plan for AI Agent Memory System with Graphiti MCP

Based on the Graphiti MCP implementation, here's a comprehensive guide for implementing an effective memory system for your AI agent:

## 1. Prompt Patterns for Storing Problem-Solving Experiences

### A. Post-Problem-Solving Storage Prompts

Here are specific prompts you should use after your agent completes a task:

```
MEMORY_STORAGE_PROMPT = """
I have just completed a problem-solving task. I need to store this experience in my memory system using Graphiti MCP for future reference.

Task Details:
- Original Problem: {original_problem}
- Solution Approach: {solution_approach}
- Key Insights: {key_insights}
- Mistakes Made: {mistakes_made}
- Final Outcome: {final_outcome}
- Context/Domain: {context_domain}

Please use the add_memory tool to store this experience with the following structure:

1. **Problem Summary Episode**: Store the core problem and its context
2. **Solution Process Episode**: Store the step-by-step approach taken
3. **Insights and Lessons Episode**: Store key learnings and insights
4. **Mistakes and Corrections Episode**: Store what went wrong and how it was fixed

Use descriptive names and include relevant tags in the episode content for better retrieval.
"""
```

### B. Structured Episode Content Templates

```python
# Example episode structures for different types of memories

# 1. Problem Definition Episode
problem_episode = {
    "name": "Problem: {problem_type} - {brief_description}",
    "episode_body": """
    PROBLEM TYPE: {problem_type}
    DOMAIN: {domain}
    DESCRIPTION: {detailed_problem_description}
    CONSTRAINTS: {constraints_and_limitations}
    SUCCESS_CRITERIA: {what_constitutes_success}
    CONTEXT: {relevant_background_context}
    TAGS: problem, {domain}, {problem_type}, {difficulty_level}
    """,
    "source": "text",
    "source_description": "problem_definition"
}

# 2. Solution Process Episode
solution_episode = {
    "name": "Solution: {problem_type} - {approach_name}",
    "episode_body": """
    APPROACH: {solution_approach_name}
    STEPS_TAKEN:
    1. {step_1}
    2. {step_2}
    3. {step_3}
    ...

    TOOLS_USED: {tools_and_technologies}
    DECISION_POINTS: {key_decisions_made}
    ALTERNATIVE_APPROACHES_CONSIDERED: {other_options}
    TIME_TAKEN: {duration}
    EFFECTIVENESS: {how_well_it_worked}
    TAGS: solution, {domain}, {approach_type}, {tools_used}
    """,
    "source": "text",
    "source_description": "solution_process"
}

# 3. Insights and Lessons Episode
insights_episode = {
    "name": "Insights: {problem_type} - {key_insight}",
    "episode_body": """
    KEY_INSIGHTS:
    - {insight_1}
    - {insight_2}
    - {insight_3}

    LESSONS_LEARNED:
    - {lesson_1}
    - {lesson_2}

    PATTERNS_IDENTIFIED: {recurring_patterns}
    BEST_PRACTICES: {what_worked_well}
    FUTURE_APPLICATIONS: {where_this_applies}
    TAGS: insights, lessons, {domain}, {pattern_type}
    """,
    "source": "text",
    "source_description": "insights_and_lessons"
}

# 4. Mistakes and Corrections Episode
mistakes_episode = {
    "name": "Mistakes: {problem_type} - {mistake_category}",
    "episode_body": """
    MISTAKES_MADE:
    - {mistake_1}: {description_and_impact}
    - {mistake_2}: {description_and_impact}

    ROOT_CAUSES:
    - {cause_1}
    - {cause_2}

    CORRECTIONS_APPLIED:
    - {correction_1}
    - {correction_2}

    PREVENTION_STRATEGIES: {how_to_avoid_in_future}
    WARNING_SIGNS: {early_indicators_of_this_mistake}
    TAGS: mistakes, corrections, {domain}, {mistake_type}
    """,
    "source": "text",
    "source_description": "mistakes_and_corrections"
}
```

## 2. Retrieval Prompts for Applying Past Experiences

### A. Pre-Task Memory Retrieval Prompt

```
MEMORY_RETRIEVAL_PROMPT = """
I am about to work on a new task: {new_task_description}

Domain: {task_domain}
Type: {task_type}
Context: {current_context}

Before I begin, I need to search my memory for relevant past experiences that could help me approach this task more effectively.

Please use the search_memory_nodes and search_memory_facts tools to find:

1. **Similar Problems**: Search for "problem {task_type} {domain} {key_keywords}"
2. **Relevant Solutions**: Search for "solution {approach_type} {tools_needed}"
3. **Related Insights**: Search for "insights {domain} {pattern_keywords}"
4. **Common Mistakes**: Search for "mistakes {task_type} {domain}"

Prioritize memories that are:
- From the same or similar domain
- Involving similar problem types
- Recent and successful
- Containing specific actionable insights
"""
```

### B. Dynamic Memory Integration Prompt

```
MEMORY_INTEGRATION_PROMPT = """
Based on the retrieved memories, I will now integrate past learnings into my current approach:

Retrieved Memories Summary:
{memory_summary}

Current Task: {current_task}

Integration Strategy:
1. **Apply Successful Patterns**: {patterns_to_apply}
2. **Avoid Known Mistakes**: {mistakes_to_avoid}
3. **Use Proven Tools/Approaches**: {tools_and_approaches}
4. **Adapt Insights**: {how_to_adapt_insights}

Modified Approach:
{updated_approach_based_on_memory}
"""
```

## 3. Best Practices for Categorization and Tagging

### A. Hierarchical Tagging System

```python
# Recommended tagging structure for problem-solving memories

TAGGING_SCHEMA = {
    "primary_category": [
        "problem", "solution", "insight", "mistake", "procedure"
    ],
    "domain": [
        "coding", "debugging", "architecture", "testing", "deployment",
        "data_analysis", "ml_training", "api_integration", "database"
    ],
    "complexity": [
        "simple", "moderate", "complex", "expert"
    ],
    "outcome": [
        "successful", "partial", "failed", "abandoned"
    ],
    "tools": [
        "python", "javascript", "sql", "docker", "kubernetes", "aws"
    ],
    "pattern_type": [
        "architectural", "algorithmic", "debugging", "optimization"
    ]
}

# Example of well-tagged episode content:
tagged_episode_body = """
PROBLEM: Database connection timeout in production environment
DOMAIN: backend_development
COMPLEXITY: moderate
TOOLS: postgresql, python, sqlalchemy
PATTERN_TYPE: debugging

DESCRIPTION: Application experiencing intermittent database connection timeouts during peak traffic hours...

TAGS: problem, backend_development, database, postgresql, timeout, production, debugging, moderate
"""
```

### B. Custom Entity Types for Problem-Solving

```python
# Custom entity types for problem-solving memories
class ProblemSolution(BaseModel):
    """A ProblemSolution represents a successful approach to solving a specific type of problem."""

    problem_type: str = Field(
        ...,
        description="The category/type of problem this solution addresses"
    )
    domain: str = Field(
        ...,
        description="The domain or field where this solution applies"
    )
    approach: str = Field(
        ...,
        description="Brief description of the solution approach"
    )
    effectiveness: str = Field(
        ...,
        description="How effective this solution was (high/medium/low)"
    )

class LessonLearned(BaseModel):
    """A LessonLearned captures important insights from problem-solving experiences."""

    context: str = Field(
        ...,
        description="The context or situation where this lesson was learned"
    )
    insight: str = Field(
        ...,
        description="The key insight or lesson learned"
    )
    applicability: str = Field(
        ...,
        description="Where and when this lesson can be applied"
    )

class CommonMistake(BaseModel):
    """A CommonMistake represents a frequently made error and how to avoid it."""

    mistake_type: str = Field(
        ...,
        description="The category of mistake"
    )
    description: str = Field(
        ...,
        description="Description of the mistake and its consequences"
    )
    prevention: str = Field(
        ...,
        description="How to prevent or avoid this mistake"
    )
```

## 4. Concrete Implementation Examples

### A. Storing a Debugging Experience

```python
# Example: Storing a debugging session memory
debugging_memory = {
    "name": "Debug: API Timeout Issue - Connection Pool Exhaustion",
    "episode_body": """
    PROBLEM_TYPE: performance_debugging
    DOMAIN: backend_api
    DESCRIPTION: API endpoints returning 504 timeout errors during peak traffic

    INVESTIGATION_STEPS:
    1. Checked server logs - found connection pool warnings
    2. Monitored database connections - pool exhaustion confirmed
    3. Analyzed connection lifecycle - connections not being released
    4. Identified root cause - missing connection.close() in error handlers

    SOLUTION: Added proper connection cleanup in try/finally blocks
    TOOLS_USED: postgresql_logs, application_monitoring, connection_profiler
    TIME_TO_RESOLVE: 3 hours
    EFFECTIVENESS: high - completely resolved the issue

    KEY_INSIGHTS:
    - Always check connection pool metrics first for timeout issues
    - Error handlers must include resource cleanup
    - Connection pool monitoring is critical for production APIs

    TAGS: debugging, performance, api, database, connection_pool, timeout, backend
    """,
    "source": "text",
    "source_description": "debugging_session"
}
```

### B. Retrieving Relevant Memories

```python
# Example: Searching for relevant debugging experiences
search_queries = [
    "debugging api timeout performance database",
    "connection pool exhaustion postgresql",
    "backend performance issues production",
    "mistakes timeout debugging database"
]

# The agent would use these queries with search_memory_nodes and search_memory_facts
```

## 5. Memory-Driven Problem-Solving Workflow

Here's a complete workflow your agent should follow:

```
MEMORY_ENHANCED_WORKFLOW = """
1. **Pre-Task Memory Search**:
   - Search for similar problems: search_memory_nodes("problem {domain} {type}")
   - Search for relevant solutions: search_memory_facts("solution {keywords}")
   - Search for common mistakes: search_memory_nodes("mistakes {domain}")

2. **Memory Integration**:
   - Analyze retrieved memories for applicable patterns
   - Identify potential pitfalls to avoid
   - Select proven tools and approaches
   - Adapt successful strategies to current context

3. **Execute with Memory Guidance**:
   - Apply learned patterns and best practices
   - Monitor for warning signs of known mistakes
   - Use proven tools and approaches when applicable

4. **Post-Task Memory Storage**:
   - Store problem definition and context
   - Store solution process and decisions made
   - Store insights and lessons learned
   - Store any mistakes made and corrections applied
   - Tag everything appropriately for future retrieval

5. **Memory Refinement**:
   - Update existing memories if new insights contradict old ones
   - Create connections between related problem-solving experiences
   - Identify emerging patterns across multiple experiences
"""
```

## 6. Available Graphiti MCP Tools

The Graphiti MCP server provides these key tools for memory management:

- **add_memory**: Add episodes to the knowledge graph (supports text, JSON, and message formats)
- **search_memory_nodes**: Search for relevant node summaries using natural language queries
- **search_memory_facts**: Search for relevant facts (relationships between entities)
- **get_episodes**: Retrieve recent episodes for a specific group
- **delete_entity_edge**: Remove specific relationships from memory
- **clear_graph**: Reset the entire memory system

## 7. Implementation Tips

1. **Consistent Tagging**: Always use consistent tag formats for better retrieval
2. **Descriptive Names**: Use clear, searchable names for episodes
3. **Group Organization**: Use group_ids to separate different types of memories or projects
4. **Regular Retrieval**: Always search memory before starting new tasks
5. **Iterative Improvement**: Continuously refine your tagging and storage strategies based on retrieval effectiveness

This comprehensive approach will give your AI agent a robust memory system that continuously improves its problem-solving capabilities by learning from past experiences. The key is consistent use of structured storage and retrieval patterns, along with proper tagging for effective memory organization.
