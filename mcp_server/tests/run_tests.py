#!/usr/bin/env python3
"""
Test runner script for MCP server memory system tests.
Provides different test execution modes and reporting.
"""

import argparse
import asyncio
import os
import subprocess
import sys
from pathlib import Path
from typing import List, Optional

# Add the mcp_server directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


class TestRunner:
    """Test runner for MCP server memory system."""
    
    def __init__(self):
        self.test_dir = Path(__file__).parent
        self.project_dir = self.test_dir.parent
        
    def run_command(self, cmd: List[str], cwd: Optional[Path] = None) -> int:
        """Run a command and return exit code."""
        if cwd is None:
            cwd = self.project_dir
            
        print(f"Running: {' '.join(cmd)}")
        print(f"Working directory: {cwd}")
        
        result = subprocess.run(cmd, cwd=cwd)
        return result.returncode
    
    def install_dependencies(self) -> int:
        """Install test dependencies."""
        print("Installing test dependencies...")
        return self.run_command(["pip", "install", "-e", ".[test]"])
    
    def run_unit_tests(self, verbose: bool = False) -> int:
        """Run unit tests only."""
        print("\n" + "="*50)
        print("RUNNING UNIT TESTS")
        print("="*50)
        
        cmd = ["python", "-m", "pytest", "-m", "unit"]
        if verbose:
            cmd.append("-v")
        cmd.extend(["tests/test_memory_entity_types.py", "tests/test_memory_enhanced_server.py"])
        
        return self.run_command(cmd)
    
    def run_integration_tests(self, verbose: bool = False) -> int:
        """Run integration tests."""
        print("\n" + "="*50)
        print("RUNNING INTEGRATION TESTS")
        print("="*50)
        print("Note: These tests require a running Neo4j instance and API keys")
        
        cmd = ["python", "-m", "pytest", "-m", "integration"]
        if verbose:
            cmd.append("-v")
        cmd.append("tests/test_integration.py")
        
        return self.run_command(cmd)
    
    def run_e2e_tests(self, verbose: bool = False) -> int:
        """Run end-to-end tests."""
        print("\n" + "="*50)
        print("RUNNING END-TO-END TESTS")
        print("="*50)
        print("Note: These tests may require a running MCP server")
        
        cmd = ["python", "-m", "pytest", "-m", "e2e"]
        if verbose:
            cmd.append("-v")
        cmd.append("tests/test_e2e_mcp.py")
        
        return self.run_command(cmd)
    
    def run_all_tests(self, verbose: bool = False) -> int:
        """Run all tests."""
        print("\n" + "="*50)
        print("RUNNING ALL TESTS")
        print("="*50)
        
        cmd = ["python", "-m", "pytest"]
        if verbose:
            cmd.append("-v")
        cmd.append("tests/")
        
        return self.run_command(cmd)
    
    def run_fast_tests(self, verbose: bool = False) -> int:
        """Run only fast tests (unit tests)."""
        print("\n" + "="*50)
        print("RUNNING FAST TESTS (UNIT ONLY)")
        print("="*50)
        
        return self.run_unit_tests(verbose)
    
    def run_coverage_report(self) -> int:
        """Run tests with coverage reporting."""
        print("\n" + "="*50)
        print("RUNNING TESTS WITH COVERAGE")
        print("="*50)
        
        # Install coverage if not available
        subprocess.run(["pip", "install", "coverage"], cwd=self.project_dir)
        
        # Run tests with coverage
        cmd = [
            "python", "-m", "coverage", "run", "-m", "pytest", 
            "-m", "unit", "tests/"
        ]
        result = self.run_command(cmd)
        
        if result == 0:
            # Generate coverage report
            print("\nGenerating coverage report...")
            self.run_command(["python", "-m", "coverage", "report"])
            self.run_command(["python", "-m", "coverage", "html"])
            print("HTML coverage report generated in htmlcov/")
        
        return result
    
    def check_environment(self) -> bool:
        """Check if the test environment is properly set up."""
        print("Checking test environment...")
        
        # Check if required files exist
        required_files = [
            "memory_enhanced_server.py",
            "memory_entity_types.py",
            "graphiti_mcp_server.py"
        ]
        
        missing_files = []
        for file in required_files:
            if not (self.project_dir / file).exists():
                missing_files.append(file)
        
        if missing_files:
            print(f"Missing required files: {missing_files}")
            return False
        
        # Check if test files exist
        test_files = [
            "tests/test_memory_entity_types.py",
            "tests/test_memory_enhanced_server.py",
            "tests/test_integration.py",
            "tests/test_e2e_mcp.py"
        ]
        
        missing_test_files = []
        for file in test_files:
            if not (self.project_dir / file).exists():
                missing_test_files.append(file)
        
        if missing_test_files:
            print(f"Missing test files: {missing_test_files}")
            return False
        
        print("Environment check passed!")
        return True
    
    def print_test_summary(self):
        """Print a summary of available tests."""
        print("\n" + "="*60)
        print("MCP SERVER MEMORY SYSTEM TEST SUITE")
        print("="*60)
        print("""
Available test categories:

1. UNIT TESTS (--unit)
   - Test custom entity types validation
   - Test memory storage/retrieval functions in isolation
   - Test error handling and edge cases
   - Fast execution, no external dependencies

2. INTEGRATION TESTS (--integration)
   - Test with real Graphiti client and Neo4j
   - Test end-to-end memory workflows
   - Requires: Neo4j running, API keys configured
   - Moderate execution time

3. END-TO-END TESTS (--e2e)
   - Test MCP protocol compliance
   - Test real client-server interactions
   - Test performance characteristics
   - Requires: Full system setup
   - Slower execution

4. ALL TESTS (--all)
   - Runs all test categories
   - Comprehensive validation
   - Requires: Full environment setup

5. FAST TESTS (--fast)
   - Same as unit tests
   - Quick validation during development

6. COVERAGE (--coverage)
   - Runs unit tests with coverage reporting
   - Generates HTML coverage report
        """)


def main():
    """Main test runner function."""
    parser = argparse.ArgumentParser(
        description="Test runner for MCP server memory system",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--unit", action="store_true", help="Run unit tests only")
    parser.add_argument("--integration", action="store_true", help="Run integration tests")
    parser.add_argument("--e2e", action="store_true", help="Run end-to-end tests")
    parser.add_argument("--all", action="store_true", help="Run all tests")
    parser.add_argument("--fast", action="store_true", help="Run fast tests (unit only)")
    parser.add_argument("--coverage", action="store_true", help="Run tests with coverage")
    parser.add_argument("--install", action="store_true", help="Install test dependencies")
    parser.add_argument("--check", action="store_true", help="Check test environment")
    parser.add_argument("--summary", action="store_true", help="Show test summary")
    parser.add_argument("-v", "--verbose", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    runner = TestRunner()
    
    # Show summary if no specific action requested
    if not any([args.unit, args.integration, args.e2e, args.all, args.fast, 
                args.coverage, args.install, args.check]):
        args.summary = True
    
    if args.summary:
        runner.print_test_summary()
        return 0
    
    if args.check:
        if runner.check_environment():
            print("Environment check passed!")
            return 0
        else:
            print("Environment check failed!")
            return 1
    
    if args.install:
        return runner.install_dependencies()
    
    # Check environment before running tests
    if not runner.check_environment():
        print("Environment check failed. Run with --check for details.")
        return 1
    
    exit_code = 0
    
    if args.unit:
        exit_code = runner.run_unit_tests(args.verbose)
    elif args.integration:
        exit_code = runner.run_integration_tests(args.verbose)
    elif args.e2e:
        exit_code = runner.run_e2e_tests(args.verbose)
    elif args.all:
        exit_code = runner.run_all_tests(args.verbose)
    elif args.fast:
        exit_code = runner.run_fast_tests(args.verbose)
    elif args.coverage:
        exit_code = runner.run_coverage_report()
    
    if exit_code == 0:
        print("\n✅ Tests completed successfully!")
    else:
        print(f"\n❌ Tests failed with exit code {exit_code}")
    
    return exit_code


if __name__ == "__main__":
    sys.exit(main())
