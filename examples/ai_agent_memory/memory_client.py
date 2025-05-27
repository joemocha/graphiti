"""
AI Agent Memory Client - Client-side implementation for memory-driven problem solving.

This module provides a client interface for AI agents to interact with the Graphiti
memory system following the patterns outlined in PLAN.md.
"""

import asyncio
import json
import logging
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

# This would be your MCP client implementation
# For this example, we'll simulate MCP calls
class MCPClient:
    """Mock MCP client for demonstration. Replace with actual MCP client implementation."""
    
    def __init__(self, server_url: str = "http://localhost:8001"):
        self.server_url = server_url
        self.logger = logging.getLogger(__name__)
    
    async def call_tool(self, tool_name: str, **kwargs) -> Dict[str, Any]:
        """Simulate MCP tool call. Replace with actual MCP client call."""
        self.logger.info(f"Calling tool: {tool_name} with args: {kwargs}")
        # In real implementation, this would make actual MCP calls
        return {"message": f"Simulated call to {tool_name}", "success": True}


class AIAgentMemoryClient:
    """
    Client for AI agents to interact with Graphiti memory system.
    Implements the memory-driven problem-solving workflow from PLAN.md.
    """
    
    def __init__(self, mcp_client: MCPClient, agent_id: str = "default_agent"):
        self.mcp_client = mcp_client
        self.agent_id = agent_id
        self.logger = logging.getLogger(__name__)
        
    async def pre_task_memory_search(
        self, 
        task_description: str,
        task_domain: str,
        task_type: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Search memory for relevant past experiences before starting a task.
        Implements the Pre-Task Memory Retrieval pattern from PLAN.md.
        """
        self.logger.info(f"Searching memory for task: {task_description}")
        
        memory_results = {
            "similar_problems": [],
            "relevant_solutions": [],
            "related_insights": [],
            "common_mistakes": []
        }
        
        try:
            # 1. Search for similar problems
            similar_problems = await self.mcp_client.call_tool(
                "retrieve_similar_problems",
                current_problem=task_description,
                domain=task_domain,
                max_results=5
            )
            memory_results["similar_problems"] = similar_problems.get("nodes", [])
            
            # 2. Search for relevant solutions
            solution_query = f"solution {task_type} {task_domain}"
            relevant_solutions = await self.mcp_client.call_tool(
                "search_memory_facts",
                query=solution_query,
                max_facts=5
            )
            memory_results["relevant_solutions"] = relevant_solutions.get("facts", [])
            
            # 3. Search for related insights
            insights_query = f"insights {task_domain} {task_type}"
            related_insights = await self.mcp_client.call_tool(
                "get_lessons_for_domain",
                domain=task_domain,
                max_lessons=5
            )
            memory_results["related_insights"] = related_insights.get("facts", [])
            
            # 4. Search for common mistakes
            mistakes_query = f"mistakes {task_type} {task_domain}"
            common_mistakes = await self.mcp_client.call_tool(
                "search_memory_nodes",
                query=mistakes_query,
                max_nodes=5
            )
            memory_results["common_mistakes"] = common_mistakes.get("nodes", [])
            
            self.logger.info("Memory search completed successfully")
            return memory_results
            
        except Exception as e:
            self.logger.error(f"Error during memory search: {e}")
            return memory_results
    
    async def integrate_memory_into_approach(
        self,
        task_description: str,
        memory_results: Dict[str, Any],
        initial_approach: str
    ) -> Dict[str, str]:
        """
        Integrate retrieved memories into the problem-solving approach.
        Implements the Memory Integration pattern from PLAN.md.
        """
        integration_strategy = {
            "patterns_to_apply": "",
            "mistakes_to_avoid": "",
            "tools_and_approaches": "",
            "adapted_insights": "",
            "modified_approach": initial_approach
        }
        
        try:
            # Extract patterns from similar problems
            if memory_results["similar_problems"]:
                patterns = []
                for problem in memory_results["similar_problems"]:
                    if "summary" in problem:
                        patterns.append(problem["summary"])
                integration_strategy["patterns_to_apply"] = "; ".join(patterns)
            
            # Extract mistakes to avoid
            if memory_results["common_mistakes"]:
                mistakes = []
                for mistake in memory_results["common_mistakes"]:
                    if "summary" in mistake:
                        mistakes.append(mistake["summary"])
                integration_strategy["mistakes_to_avoid"] = "; ".join(mistakes)
            
            # Extract proven tools and approaches
            if memory_results["relevant_solutions"]:
                tools = []
                for solution in memory_results["relevant_solutions"]:
                    if "fact" in solution:
                        tools.append(solution["fact"])
                integration_strategy["tools_and_approaches"] = "; ".join(tools)
            
            # Extract and adapt insights
            if memory_results["related_insights"]:
                insights = []
                for insight in memory_results["related_insights"]:
                    if "fact" in insight:
                        insights.append(insight["fact"])
                integration_strategy["adapted_insights"] = "; ".join(insights)
            
            # Modify approach based on memory
            modified_approach = self._modify_approach_with_memory(
                initial_approach, integration_strategy
            )
            integration_strategy["modified_approach"] = modified_approach
            
            return integration_strategy
            
        except Exception as e:
            self.logger.error(f"Error integrating memory: {e}")
            return integration_strategy
    
    def _modify_approach_with_memory(
        self, 
        initial_approach: str, 
        integration_strategy: Dict[str, str]
    ) -> str:
        """Modify the initial approach based on memory insights."""
        
        modifications = []
        
        if integration_strategy["patterns_to_apply"]:
            modifications.append(f"Apply proven patterns: {integration_strategy['patterns_to_apply']}")
        
        if integration_strategy["mistakes_to_avoid"]:
            modifications.append(f"Avoid known mistakes: {integration_strategy['mistakes_to_avoid']}")
        
        if integration_strategy["tools_and_approaches"]:
            modifications.append(f"Use proven tools: {integration_strategy['tools_and_approaches']}")
        
        if modifications:
            return f"{initial_approach}\n\nMemory-guided modifications:\n" + "\n".join(modifications)
        
        return initial_approach
    
    async def store_problem_solving_session(
        self,
        problem_name: str,
        problem_description: str,
        solution_approach: str,
        key_insights: str,
        mistakes_made: str = "",
        tools_used: str = "",
        domain: str = "",
        effectiveness: str = "medium",
        final_outcome: str = ""
    ) -> bool:
        """
        Store a complete problem-solving session in memory.
        Implements the Post-Task Memory Storage pattern from PLAN.md.
        """
        try:
            # Store the complete experience using the enhanced MCP tool
            result = await self.mcp_client.call_tool(
                "store_problem_solving_experience",
                problem_name=problem_name,
                problem_description=problem_description,
                solution_approach=solution_approach,
                key_insights=key_insights,
                mistakes_made=mistakes_made,
                tools_used=tools_used,
                domain=domain,
                effectiveness=effectiveness,
                group_id=self.agent_id
            )
            
            if result.get("error"):
                self.logger.error(f"Error storing memory: {result['error']}")
                return False
            
            self.logger.info(f"Successfully stored problem-solving session: {problem_name}")
            return True
            
        except Exception as e:
            self.logger.error(f"Error storing problem-solving session: {e}")
            return False
    
    async def memory_driven_problem_solving(
        self,
        problem_description: str,
        domain: str,
        problem_type: str,
        initial_approach: str,
        context: str = ""
    ) -> Dict[str, Any]:
        """
        Complete memory-driven problem-solving workflow.
        Implements the full workflow from PLAN.md.
        """
        workflow_results = {
            "memory_search": {},
            "integration_strategy": {},
            "execution_guidance": {},
            "storage_success": False
        }
        
        try:
            # Step 1: Pre-task memory search
            self.logger.info("Step 1: Searching memory for relevant experiences")
            memory_results = await self.pre_task_memory_search(
                problem_description, domain, problem_type, context
            )
            workflow_results["memory_search"] = memory_results
            
            # Step 2: Memory integration
            self.logger.info("Step 2: Integrating memory into approach")
            integration_strategy = await self.integrate_memory_into_approach(
                problem_description, memory_results, initial_approach
            )
            workflow_results["integration_strategy"] = integration_strategy
            
            # Step 3: Provide execution guidance
            self.logger.info("Step 3: Generating execution guidance")
            execution_guidance = self._generate_execution_guidance(
                memory_results, integration_strategy
            )
            workflow_results["execution_guidance"] = execution_guidance
            
            return workflow_results
            
        except Exception as e:
            self.logger.error(f"Error in memory-driven problem solving: {e}")
            return workflow_results
    
    def _generate_execution_guidance(
        self,
        memory_results: Dict[str, Any],
        integration_strategy: Dict[str, str]
    ) -> Dict[str, Any]:
        """Generate execution guidance based on memory and integration strategy."""
        
        guidance = {
            "recommended_approach": integration_strategy.get("modified_approach", ""),
            "warning_signs": [],
            "success_indicators": [],
            "checkpoints": []
        }
        
        # Extract warning signs from common mistakes
        for mistake in memory_results.get("common_mistakes", []):
            if "summary" in mistake:
                guidance["warning_signs"].append(mistake["summary"])
        
        # Extract success indicators from solutions
        for solution in memory_results.get("relevant_solutions", []):
            if "fact" in solution:
                guidance["success_indicators"].append(solution["fact"])
        
        # Generate checkpoints based on insights
        for insight in memory_results.get("related_insights", []):
            if "fact" in insight:
                guidance["checkpoints"].append(f"Verify: {insight['fact']}")
        
        return guidance


# Example usage and testing
async def example_usage():
    """Example of how to use the AI Agent Memory Client."""
    
    # Initialize client
    mcp_client = MCPClient()
    memory_client = AIAgentMemoryClient(mcp_client, agent_id="example_agent")
    
    # Example problem-solving session
    problem_description = "API endpoints returning 504 timeout errors during peak traffic"
    domain = "backend_api"
    problem_type = "performance_debugging"
    initial_approach = "Check server logs and monitor database connections"
    
    # Run memory-driven problem solving
    results = await memory_client.memory_driven_problem_solving(
        problem_description=problem_description,
        domain=domain,
        problem_type=problem_type,
        initial_approach=initial_approach,
        context="Production environment with high traffic"
    )
    
    print("Memory-driven problem solving results:")
    print(json.dumps(results, indent=2))
    
    # Store the session results (after actual problem solving)
    success = await memory_client.store_problem_solving_session(
        problem_name="API Timeout Issue",
        problem_description=problem_description,
        solution_approach="Added connection pool monitoring and proper cleanup",
        key_insights="Always check connection pool metrics first for timeout issues",
        mistakes_made="Initially focused on server resources instead of connection pool",
        tools_used="postgresql_logs, application_monitoring, connection_profiler",
        domain=domain,
        effectiveness="high",
        final_outcome="Issue completely resolved, no more timeouts"
    )
    
    print(f"Memory storage success: {success}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    asyncio.run(example_usage())
