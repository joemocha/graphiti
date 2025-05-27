#!/usr/bin/env python3
"""
Quick validation script to verify the testing setup is working correctly.
This script performs basic checks without requiring external dependencies.
"""

import sys
import traceback
from pathlib import Path

# Add the mcp_server directory to Python path
sys.path.insert(0, str(Path(__file__).parent.parent))


def test_imports():
    """Test that all required modules can be imported."""
    print("Testing imports...")
    
    try:
        # Test memory entity types
        from memory_entity_types import (
            ProblemSolution,
            LessonLearned,
            CommonMistake,
            ProblemContext,
            SuccessPattern,
            MEMORY_ENTITY_TYPES,
        )
        print("✅ Memory entity types imported successfully")
        
        # Test memory enhanced server
        from memory_enhanced_server import (
            store_problem_solving_experience,
            retrieve_similar_problems,
            get_lessons_for_domain,
            ENHANCED_ENTITY_TYPES,
        )
        print("✅ Memory enhanced server imported successfully")
        
        # Test base server components
        from graphiti_mcp_server import (
            GraphitiConfig,
            ENTITY_TYPES,
            ErrorResponse,
            SuccessResponse,
        )
        print("✅ Base server components imported successfully")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        traceback.print_exc()
        return False
    except Exception as e:
        print(f"❌ Unexpected error during imports: {e}")
        traceback.print_exc()
        return False


def test_entity_creation():
    """Test that entity types can be created and validated."""
    print("\nTesting entity creation...")
    
    try:
        from memory_entity_types import (
            ProblemSolution,
            LessonLearned,
            CommonMistake,
            ProblemContext,
            SuccessPattern,
        )
        
        # Test ProblemSolution
        solution = ProblemSolution(
            problem_type="test_problem",
            domain="test_domain",
            approach="test_approach",
            effectiveness="high"
        )
        assert solution.problem_type == "test_problem"
        print("✅ ProblemSolution created successfully")
        
        # Test LessonLearned
        lesson = LessonLearned(
            context="test_context",
            insight="test_insight",
            applicability="test_applicability"
        )
        assert lesson.context == "test_context"
        print("✅ LessonLearned created successfully")
        
        # Test CommonMistake
        mistake = CommonMistake(
            mistake_type="test_mistake",
            description="test_description",
            prevention="test_prevention"
        )
        assert mistake.mistake_type == "test_mistake"
        print("✅ CommonMistake created successfully")
        
        # Test ProblemContext
        context = ProblemContext(
            problem_domain="test_domain"
        )
        assert context.problem_domain == "test_domain"
        print("✅ ProblemContext created successfully")
        
        # Test SuccessPattern
        pattern = SuccessPattern(
            pattern_name="test_pattern",
            description="test_description",
            conditions="test_conditions"
        )
        assert pattern.pattern_name == "test_pattern"
        print("✅ SuccessPattern created successfully")
        
        return True
        
    except Exception as e:
        print(f"❌ Entity creation error: {e}")
        traceback.print_exc()
        return False


def test_entity_validation():
    """Test entity validation with invalid data."""
    print("\nTesting entity validation...")
    
    try:
        from memory_entity_types import ProblemSolution
        from pydantic import ValidationError
        
        # Test that validation works for missing required fields
        try:
            ProblemSolution()  # Missing required fields
            print("❌ Validation should have failed for missing required fields")
            return False
        except ValidationError:
            print("✅ Validation correctly rejected missing required fields")
        
        # Test that validation works for valid data
        try:
            solution = ProblemSolution(
                problem_type="valid_type",
                domain="valid_domain", 
                approach="valid_approach",
                effectiveness="high"
            )
            print("✅ Validation correctly accepted valid data")
        except ValidationError as e:
            print(f"❌ Validation incorrectly rejected valid data: {e}")
            return False
        
        return True
        
    except Exception as e:
        print(f"❌ Validation testing error: {e}")
        traceback.print_exc()
        return False


def test_serialization():
    """Test that entities can be serialized to dictionaries."""
    print("\nTesting serialization...")
    
    try:
        from memory_entity_types import ProblemSolution
        
        solution = ProblemSolution(
            problem_type="serialization_test",
            domain="test_domain",
            approach="test_approach", 
            effectiveness="medium",
            tools_used="test_tools",
            complexity="simple"
        )
        
        # Test serialization
        serialized = solution.model_dump()
        assert isinstance(serialized, dict)
        assert serialized["problem_type"] == "serialization_test"
        assert serialized["domain"] == "test_domain"
        assert serialized["effectiveness"] == "medium"
        print("✅ Entity serialization works correctly")
        
        # Test that all values are JSON-serializable types
        import json
        json_str = json.dumps(serialized)
        assert isinstance(json_str, str)
        print("✅ Serialized entity is JSON-compatible")
        
        return True
        
    except Exception as e:
        print(f"❌ Serialization error: {e}")
        traceback.print_exc()
        return False


def test_enhanced_entity_types():
    """Test that ENHANCED_ENTITY_TYPES includes all expected types."""
    print("\nTesting enhanced entity types configuration...")
    
    try:
        from memory_enhanced_server import ENHANCED_ENTITY_TYPES
        from memory_entity_types import MEMORY_ENTITY_TYPES
        from graphiti_mcp_server import ENTITY_TYPES
        
        # Check that memory types are included
        for name, entity_type in MEMORY_ENTITY_TYPES.items():
            assert name in ENHANCED_ENTITY_TYPES
            assert ENHANCED_ENTITY_TYPES[name] == entity_type
        print("✅ All memory entity types included in ENHANCED_ENTITY_TYPES")
        
        # Check that base types are included
        for name, entity_type in ENTITY_TYPES.items():
            assert name in ENHANCED_ENTITY_TYPES
            assert ENHANCED_ENTITY_TYPES[name] == entity_type
        print("✅ All base entity types included in ENHANCED_ENTITY_TYPES")
        
        # Check expected memory types
        expected_memory_types = {
            "ProblemSolution",
            "LessonLearned",
            "CommonMistake", 
            "ProblemContext",
            "SuccessPattern"
        }
        
        for expected_type in expected_memory_types:
            assert expected_type in ENHANCED_ENTITY_TYPES
        print("✅ All expected memory entity types are present")
        
        return True
        
    except Exception as e:
        print(f"❌ Enhanced entity types error: {e}")
        traceback.print_exc()
        return False


def test_function_signatures():
    """Test that memory functions have correct signatures."""
    print("\nTesting function signatures...")
    
    try:
        from memory_enhanced_server import (
            store_problem_solving_experience,
            retrieve_similar_problems,
            get_lessons_for_domain
        )
        import inspect
        
        # Test store_problem_solving_experience signature
        sig = inspect.signature(store_problem_solving_experience)
        required_params = {"problem_name", "problem_description", "solution_approach", "key_insights"}
        actual_params = set(sig.parameters.keys())
        assert required_params.issubset(actual_params)
        print("✅ store_problem_solving_experience has correct signature")
        
        # Test retrieve_similar_problems signature
        sig = inspect.signature(retrieve_similar_problems)
        required_params = {"current_problem"}
        actual_params = set(sig.parameters.keys())
        assert required_params.issubset(actual_params)
        print("✅ retrieve_similar_problems has correct signature")
        
        # Test get_lessons_for_domain signature
        sig = inspect.signature(get_lessons_for_domain)
        required_params = {"domain"}
        actual_params = set(sig.parameters.keys())
        assert required_params.issubset(actual_params)
        print("✅ get_lessons_for_domain has correct signature")
        
        return True
        
    except Exception as e:
        print(f"❌ Function signature error: {e}")
        traceback.print_exc()
        return False


def main():
    """Run all validation tests."""
    print("=" * 60)
    print("MCP SERVER MEMORY SYSTEM VALIDATION")
    print("=" * 60)
    
    tests = [
        test_imports,
        test_entity_creation,
        test_entity_validation,
        test_serialization,
        test_enhanced_entity_types,
        test_function_signatures,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
            else:
                failed += 1
        except Exception as e:
            print(f"❌ Test {test.__name__} failed with exception: {e}")
            failed += 1
    
    print("\n" + "=" * 60)
    print("VALIDATION SUMMARY")
    print("=" * 60)
    print(f"✅ Passed: {passed}")
    print(f"❌ Failed: {failed}")
    print(f"📊 Total: {passed + failed}")
    
    if failed == 0:
        print("\n🎉 All validation tests passed! The setup is working correctly.")
        print("\nNext steps:")
        print("1. Run unit tests: python tests/run_tests.py --unit")
        print("2. Set up Neo4j for integration tests")
        print("3. Configure API keys for full testing")
        return 0
    else:
        print(f"\n⚠️  {failed} validation test(s) failed. Please fix the issues before proceeding.")
        print("\nTroubleshooting:")
        print("1. Check that all files are in the correct locations")
        print("2. Verify Python path and imports")
        print("3. Check for syntax errors in the code")
        return 1


if __name__ == "__main__":
    sys.exit(main())
