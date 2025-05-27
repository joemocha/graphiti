#!/usr/bin/env python3
"""
Setup script for AI Agent Memory System.
This script helps initialize and configure the memory system.
"""

import asyncio
import json
import logging
import os
import sys
from pathlib import Path

# Add parent directories to path for imports
sys.path.append(str(Path(__file__).parent.parent.parent))

from graphiti_core import Graphiti
from graphiti_core.utils.maintenance.graph_data_operations import clear_data

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class MemorySystemSetup:
    """Setup and configuration for the AI Agent Memory System."""
    
    def __init__(self):
        self.neo4j_uri = os.environ.get('NEO4J_URI', 'bolt://localhost:7687')
        self.neo4j_user = os.environ.get('NEO4J_USER', 'neo4j')
        self.neo4j_password = os.environ.get('NEO4J_PASSWORD', 'password')
        self.openai_api_key = os.environ.get('OPENAI_API_KEY')
        
    async def check_prerequisites(self) -> bool:
        """Check if all prerequisites are met."""
        logger.info("Checking prerequisites...")
        
        issues = []
        
        # Check environment variables
        if not self.openai_api_key:
            issues.append("OPENAI_API_KEY environment variable not set")
        
        # Check Neo4j connection
        try:
            client = Graphiti(self.neo4j_uri, self.neo4j_user, self.neo4j_password)
            # Test connection by trying to execute a simple query
            await client.driver.execute_query("RETURN 1")
            logger.info("✓ Neo4j connection successful")
        except Exception as e:
            issues.append(f"Neo4j connection failed: {e}")
        
        if issues:
            logger.error("Prerequisites check failed:")
            for issue in issues:
                logger.error(f"  - {issue}")
            return False
        
        logger.info("✓ All prerequisites met")
        return True
    
    async def initialize_database(self, clear_existing: bool = False) -> bool:
        """Initialize the database with required indices and constraints."""
        logger.info("Initializing database...")
        
        try:
            client = Graphiti(self.neo4j_uri, self.neo4j_user, self.neo4j_password)
            
            if clear_existing:
                logger.warning("Clearing existing data...")
                await clear_data(client.driver)
            
            # Build indices and constraints
            await client.build_indices_and_constraints()
            logger.info("✓ Database initialized successfully")
            return True
            
        except Exception as e:
            logger.error(f"Database initialization failed: {e}")
            return False
    
    def create_mcp_config(self, config_type: str = "stdio") -> dict:
        """Create MCP configuration for different clients."""
        
        base_config = {
            "transport": config_type,
            "command": "python",
            "args": [
                str(Path(__file__).parent.parent.parent / "mcp_server" / "memory_enhanced_server.py"),
                "--transport", config_type,
                "--use-custom-entities",
                "--group-id", "memory_agent"
            ],
            "env": {
                "NEO4J_URI": self.neo4j_uri,
                "NEO4J_USER": self.neo4j_user,
                "NEO4J_PASSWORD": self.neo4j_password,
                "OPENAI_API_KEY": "${OPENAI_API_KEY}"
            }
        }
        
        if config_type == "stdio":
            return {
                "mcpServers": {
                    "graphiti-memory": base_config
                }
            }
        else:  # sse
            base_config["host"] = "localhost"
            base_config["port"] = 8001
            return {
                "mcpServers": {
                    "graphiti-memory": base_config
                }
            }
    
    def save_config_file(self, config_type: str = "stdio", output_path: str = None):
        """Save MCP configuration to file."""
        
        if output_path is None:
            output_path = f"mcp_config_{config_type}.json"
        
        config = self.create_mcp_config(config_type)
        
        with open(output_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        logger.info(f"✓ MCP configuration saved to {output_path}")
        return output_path
    
    async def create_sample_memories(self) -> bool:
        """Create sample memories for testing and demonstration."""
        logger.info("Creating sample memories...")
        
        try:
            # Import memory client
            from memory_client import AIAgentMemoryClient, MCPClient
            
            # Initialize clients
            mcp_client = MCPClient()
            memory_client = AIAgentMemoryClient(mcp_client, agent_id="setup_agent")
            
            # Sample problem-solving experiences
            sample_experiences = [
                {
                    "problem_name": "Database Connection Pool Exhaustion",
                    "problem_description": "Application experiencing connection timeouts during peak traffic due to exhausted database connection pool",
                    "solution_approach": "Implemented connection pool monitoring, added proper connection cleanup in error handlers, and increased pool size",
                    "key_insights": "Always monitor connection pool metrics for timeout issues; Error handlers must include resource cleanup; Connection pool size should scale with expected concurrent users",
                    "mistakes_made": "Initially focused on server CPU and memory instead of connection pool; Forgot to add cleanup in exception handlers",
                    "tools_used": "postgresql_logs, application_monitoring, connection_profiler, database_metrics",
                    "domain": "backend_database",
                    "effectiveness": "high"
                },
                {
                    "problem_name": "Frontend Performance Optimization",
                    "problem_description": "React application showing slow initial load times and poor Core Web Vitals scores",
                    "solution_approach": "Implemented code splitting, lazy loading, image optimization, and service worker caching",
                    "key_insights": "Bundle size analysis should be first step; Lazy loading has biggest impact on initial load; Service workers dramatically improve repeat visits",
                    "mistakes_made": "Over-optimized images causing quality issues; Aggressive code splitting broke some user flows",
                    "tools_used": "webpack_bundle_analyzer, lighthouse, chrome_devtools, web_vitals",
                    "domain": "frontend_performance",
                    "effectiveness": "high"
                },
                {
                    "problem_name": "API Rate Limiting Implementation",
                    "problem_description": "Need to implement rate limiting for public API to prevent abuse while maintaining good user experience",
                    "solution_approach": "Implemented sliding window rate limiting with Redis, added rate limit headers, and created tiered limits for different user types",
                    "key_insights": "Sliding window is more user-friendly than fixed window; Rate limit headers help clients implement proper backoff; Different user tiers need different limits",
                    "mistakes_made": "Initial implementation was too aggressive and blocked legitimate users; Didn't consider burst traffic patterns",
                    "tools_used": "redis, api_gateway, monitoring_dashboard, load_testing",
                    "domain": "backend_api",
                    "effectiveness": "medium"
                }
            ]
            
            # Store sample experiences
            for experience in sample_experiences:
                success = await memory_client.store_problem_solving_session(**experience)
                if success:
                    logger.info(f"✓ Stored sample memory: {experience['problem_name']}")
                else:
                    logger.warning(f"Failed to store: {experience['problem_name']}")
            
            logger.info("✓ Sample memories created successfully")
            return True
            
        except Exception as e:
            logger.error(f"Failed to create sample memories: {e}")
            return False
    
    async def run_setup(self, clear_db: bool = False, create_samples: bool = True):
        """Run complete setup process."""
        logger.info("Starting AI Agent Memory System setup...")
        
        # Check prerequisites
        if not await self.check_prerequisites():
            logger.error("Setup failed: Prerequisites not met")
            return False
        
        # Initialize database
        if not await self.initialize_database(clear_existing=clear_db):
            logger.error("Setup failed: Database initialization failed")
            return False
        
        # Create configuration files
        self.save_config_file("stdio", "mcp_config_stdio.json")
        self.save_config_file("sse", "mcp_config_sse.json")
        
        # Create sample memories
        if create_samples:
            await self.create_sample_memories()
        
        logger.info("✓ Setup completed successfully!")
        
        # Print next steps
        self._print_next_steps()
        
        return True
    
    def _print_next_steps(self):
        """Print next steps for the user."""
        print("\n" + "="*60)
        print("AI Agent Memory System Setup Complete!")
        print("="*60)
        print("\nNext Steps:")
        print("1. Start the enhanced MCP server:")
        print("   python mcp_server/memory_enhanced_server.py --use-custom-entities")
        print("\n2. Test the memory client:")
        print("   python examples/ai_agent_memory/memory_client.py")
        print("\n3. Try the LangGraph integration:")
        print("   python examples/ai_agent_memory/langgraph_memory_agent.py")
        print("\n4. For Cursor IDE integration:")
        print("   Use the generated mcp_config_stdio.json file")
        print("\n5. For other MCP clients:")
        print("   Use the generated mcp_config_sse.json file")
        print("\nConfiguration files created:")
        print("- mcp_config_stdio.json (for Cursor IDE)")
        print("- mcp_config_sse.json (for other clients)")
        print("\nDocumentation:")
        print("- See examples/ai_agent_memory/README.md for detailed usage")
        print("- See PLAN.md for implementation patterns")
        print("="*60)


async def main():
    """Main setup function."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Setup AI Agent Memory System")
    parser.add_argument("--clear-db", action="store_true", 
                       help="Clear existing database (WARNING: destructive)")
    parser.add_argument("--no-samples", action="store_true",
                       help="Skip creating sample memories")
    parser.add_argument("--check-only", action="store_true",
                       help="Only check prerequisites, don't setup")
    
    args = parser.parse_args()
    
    setup = MemorySystemSetup()
    
    if args.check_only:
        success = await setup.check_prerequisites()
        sys.exit(0 if success else 1)
    
    success = await setup.run_setup(
        clear_db=args.clear_db,
        create_samples=not args.no_samples
    )
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    asyncio.run(main())
