"""
LangGraph Agent with Graphiti Memory Integration

This example shows how to integrate the AI Agent Memory System with a LangGraph agent,
extending the pattern from examples/langgraph-agent/agent.ipynb with memory capabilities.
"""

import asyncio
import logging
import os
from datetime import datetime, timezone
from typing import Annotated, TypedDict

from dotenv import load_dotenv
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph, add_messages
from langgraph.prebuilt import ToolNode

# Import Graphiti components
from graphiti_core import Graphiti
from graphiti_core.nodes import EpisodeType

# Import our memory client
from memory_client import AIAgentMemoryClient, MCPClient

load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemoryEnhancedState(TypedDict):
    """State for memory-enhanced LangGraph agent."""
    messages: Annotated[list, add_messages]
    current_task: str
    task_domain: str
    task_type: str
    memory_context: dict
    agent_id: str


class MemoryEnhancedAgent:
    """
    LangGraph agent enhanced with Graphiti memory capabilities.
    
    This agent follows the memory-driven problem-solving workflow:
    1. Search memory before starting tasks
    2. Integrate past experiences into approach
    3. Execute with memory guidance
    4. Store results for future use
    """
    
    def __init__(self, graphiti_client: Graphiti, agent_id: str = "memory_agent"):
        self.graphiti_client = graphiti_client
        self.agent_id = agent_id
        
        # Initialize memory client (in real implementation, use actual MCP client)
        mcp_client = MCPClient()
        self.memory_client = AIAgentMemoryClient(mcp_client, agent_id)
        
        # Initialize LLM
        self.llm = ChatOpenAI(model='gpt-4.1-mini', temperature=0)
        
        # Setup tools
        self.tools = [self.search_memory_tool, self.store_memory_tool]
        self.llm_with_tools = self.llm.bind_tools(self.tools)
        self.tool_node = ToolNode(self.tools)
        
        # Build graph
        self.graph = self._build_graph()
    
    @tool
    async def search_memory_tool(self, query: str, domain: str = "", task_type: str = "") -> str:
        """Search memory for relevant past experiences."""
        try:
            memory_results = await self.memory_client.pre_task_memory_search(
                task_description=query,
                task_domain=domain,
                task_type=task_type
            )
            
            # Format results for LLM consumption
            formatted_results = []
            
            if memory_results["similar_problems"]:
                formatted_results.append("Similar Problems:")
                for problem in memory_results["similar_problems"][:3]:
                    formatted_results.append(f"- {problem.get('name', 'Unknown')}: {problem.get('summary', 'No summary')}")
            
            if memory_results["relevant_solutions"]:
                formatted_results.append("\nRelevant Solutions:")
                for solution in memory_results["relevant_solutions"][:3]:
                    formatted_results.append(f"- {solution.get('fact', 'No fact available')}")
            
            if memory_results["common_mistakes"]:
                formatted_results.append("\nCommon Mistakes to Avoid:")
                for mistake in memory_results["common_mistakes"][:3]:
                    formatted_results.append(f"- {mistake.get('summary', 'No summary')}")
            
            return "\n".join(formatted_results) if formatted_results else "No relevant memories found."
            
        except Exception as e:
            logger.error(f"Error searching memory: {e}")
            return f"Error searching memory: {e}"
    
    @tool
    async def store_memory_tool(
        self,
        problem_name: str,
        problem_description: str,
        solution: str,
        insights: str,
        domain: str = "",
        effectiveness: str = "medium"
    ) -> str:
        """Store a problem-solving experience in memory."""
        try:
            success = await self.memory_client.store_problem_solving_session(
                problem_name=problem_name,
                problem_description=problem_description,
                solution_approach=solution,
                key_insights=insights,
                domain=domain,
                effectiveness=effectiveness
            )
            
            if success:
                return f"Successfully stored memory for: {problem_name}"
            else:
                return f"Failed to store memory for: {problem_name}"
                
        except Exception as e:
            logger.error(f"Error storing memory: {e}")
            return f"Error storing memory: {e}"
    
    async def memory_enhanced_agent_node(self, state: MemoryEnhancedState):
        """Main agent node with memory enhancement."""
        
        # Extract current task information
        current_task = state.get("current_task", "")
        task_domain = state.get("task_domain", "")
        task_type = state.get("task_type", "")
        
        # If this is a new task, search memory first
        if current_task and not state.get("memory_context"):
            logger.info(f"Searching memory for task: {current_task}")
            
            try:
                memory_results = await self.memory_client.pre_task_memory_search(
                    task_description=current_task,
                    task_domain=task_domain,
                    task_type=task_type
                )
                
                # Store memory context in state
                state["memory_context"] = memory_results
                
                # Create memory-informed system message
                memory_summary = self._format_memory_for_system_message(memory_results)
                
                system_message = SystemMessage(
                    content=f"""You are an AI agent with access to memory from past problem-solving experiences.

Current Task: {current_task}
Domain: {task_domain}
Type: {task_type}

Relevant Memory Context:
{memory_summary}

Use this memory to inform your approach. Apply successful patterns, avoid known mistakes, 
and build upon previous insights. If you need to search for more specific memories or 
store new experiences, use the available tools.

Always explain how you're using past experiences to inform your current approach."""
                )
                
                messages = [system_message] + state["messages"]
                
            except Exception as e:
                logger.error(f"Error in memory search: {e}")
                messages = state["messages"]
        else:
            messages = state["messages"]
        
        # Generate response
        response = await self.llm_with_tools.ainvoke(messages)
        
        # Store interaction in Graphiti for future reference
        if len(state["messages"]) > 0:
            last_user_message = state["messages"][-1]
            asyncio.create_task(
                self.graphiti_client.add_episode(
                    name=f"Agent Interaction - {current_task[:50]}",
                    episode_body=f"User: {last_user_message.content}\nAgent: {response.content}",
                    source=EpisodeType.message,
                    reference_time=datetime.now(timezone.utc),
                    source_description="Memory Enhanced Agent",
                    group_id=self.agent_id
                )
            )
        
        return {"messages": [response]}
    
    def _format_memory_for_system_message(self, memory_results: dict) -> str:
        """Format memory results for inclusion in system message."""
        
        sections = []
        
        if memory_results.get("similar_problems"):
            sections.append("Similar Problems Solved:")
            for problem in memory_results["similar_problems"][:3]:
                sections.append(f"- {problem.get('name', 'Unknown')}")
        
        if memory_results.get("relevant_solutions"):
            sections.append("\nProven Solutions:")
            for solution in memory_results["relevant_solutions"][:3]:
                sections.append(f"- {solution.get('fact', 'No details')}")
        
        if memory_results.get("related_insights"):
            sections.append("\nKey Insights:")
            for insight in memory_results["related_insights"][:3]:
                sections.append(f"- {insight.get('fact', 'No details')}")
        
        if memory_results.get("common_mistakes"):
            sections.append("\nMistakes to Avoid:")
            for mistake in memory_results["common_mistakes"][:3]:
                sections.append(f"- {mistake.get('summary', 'No details')}")
        
        return "\n".join(sections) if sections else "No relevant memories found."
    
    def should_continue(self, state: MemoryEnhancedState):
        """Determine whether to continue to tools or end."""
        messages = state["messages"]
        last_message = messages[-1]
        
        if not last_message.tool_calls:
            return "end"
        else:
            return "continue"
    
    def _build_graph(self):
        """Build the LangGraph with memory enhancement."""
        
        graph_builder = StateGraph(MemoryEnhancedState)
        memory = MemorySaver()
        
        # Add nodes
        graph_builder.add_node("agent", self.memory_enhanced_agent_node)
        graph_builder.add_node("tools", self.tool_node)
        
        # Add edges
        graph_builder.add_edge(START, "agent")
        graph_builder.add_conditional_edges(
            "agent", 
            self.should_continue, 
            {"continue": "tools", "end": END}
        )
        graph_builder.add_edge("tools", "agent")
        
        return graph_builder.compile(checkpointer=memory)
    
    async def run_task(
        self,
        task_description: str,
        domain: str = "",
        task_type: str = "",
        user_message: str = ""
    ):
        """Run a task with memory enhancement."""
        
        initial_state = {
            "messages": [HumanMessage(content=user_message or task_description)],
            "current_task": task_description,
            "task_domain": domain,
            "task_type": task_type,
            "memory_context": {},
            "agent_id": self.agent_id
        }
        
        config = {"configurable": {"thread_id": f"{self.agent_id}_{datetime.now().isoformat()}"}}
        
        result = await self.graph.ainvoke(initial_state, config=config)
        return result


# Example usage
async def main():
    """Example of using the memory-enhanced LangGraph agent."""
    
    # Initialize Graphiti client
    neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
    neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
    neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
    
    graphiti_client = Graphiti(neo4j_uri, neo4j_user, neo4j_password)
    
    # Initialize memory-enhanced agent
    agent = MemoryEnhancedAgent(graphiti_client, agent_id="example_memory_agent")
    
    # Example task
    task_description = "Debug API performance issues in production"
    domain = "backend_development"
    task_type = "performance_debugging"
    user_message = "Our API is responding slowly during peak hours. How should I approach debugging this?"
    
    # Run the task
    result = await agent.run_task(
        task_description=task_description,
        domain=domain,
        task_type=task_type,
        user_message=user_message
    )
    
    print("Agent Response:")
    print(result["messages"][-1].content)


if __name__ == "__main__":
    asyncio.run(main())
