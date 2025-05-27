"""
Unit tests for memory entity types.
Tests validation, serialization, and deserialization of custom entity types.
"""

import pytest
from pydantic import ValidationError

from memory_entity_types import (
    ProblemSolution,
    LessonLearned,
    CommonMistake,
    ProblemContext,
    SuccessPattern,
    MEMORY_ENTITY_TYPES,
)


class TestProblemSolution:
    """Test ProblemSolution entity type."""

    @pytest.mark.unit
    def test_valid_problem_solution_creation(self, sample_entity_data):
        """Test creating a valid ProblemSolution."""
        data = sample_entity_data["problem_solution"]
        solution = ProblemSolution(**data)
        
        assert solution.problem_type == "performance_optimization"
        assert solution.domain == "backend_api"
        assert solution.approach == "Database indexing and query optimization"
        assert solution.effectiveness == "high"
        assert solution.tools_used == "PostgreSQL, pgAdmin"
        assert solution.complexity == "moderate"

    @pytest.mark.unit
    def test_problem_solution_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError) as exc_info:
            ProblemSolution()
        
        errors = exc_info.value.errors()
        required_fields = {"problem_type", "domain", "approach", "effectiveness"}
        error_fields = {error["loc"][0] for error in errors}
        
        assert required_fields.issubset(error_fields)

    @pytest.mark.unit
    def test_problem_solution_defaults(self):
        """Test default values for optional fields."""
        solution = ProblemSolution(
            problem_type="test",
            domain="test",
            approach="test",
            effectiveness="medium"
        )
        
        assert solution.tools_used == ""
        assert solution.complexity == "moderate"

    @pytest.mark.unit
    def test_problem_solution_serialization(self, sample_entity_data):
        """Test serialization to dict."""
        data = sample_entity_data["problem_solution"]
        solution = ProblemSolution(**data)
        serialized = solution.model_dump()
        
        assert isinstance(serialized, dict)
        assert serialized["problem_type"] == data["problem_type"]
        assert serialized["domain"] == data["domain"]


class TestLessonLearned:
    """Test LessonLearned entity type."""

    @pytest.mark.unit
    def test_valid_lesson_learned_creation(self, sample_entity_data):
        """Test creating a valid LessonLearned."""
        data = sample_entity_data["lesson_learned"]
        lesson = LessonLearned(**data)
        
        assert lesson.context == "API performance optimization project"
        assert lesson.insight == "Always check database performance before optimizing application code"
        assert lesson.applicability == "Any database-backed application with performance issues"
        assert lesson.confidence_level == "high"
        assert lesson.domain == "backend"

    @pytest.mark.unit
    def test_lesson_learned_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError) as exc_info:
            LessonLearned()
        
        errors = exc_info.value.errors()
        required_fields = {"context", "insight", "applicability"}
        error_fields = {error["loc"][0] for error in errors}
        
        assert required_fields.issubset(error_fields)

    @pytest.mark.unit
    def test_lesson_learned_defaults(self):
        """Test default values for optional fields."""
        lesson = LessonLearned(
            context="test",
            insight="test",
            applicability="test"
        )
        
        assert lesson.confidence_level == "medium"
        assert lesson.domain == ""


class TestCommonMistake:
    """Test CommonMistake entity type."""

    @pytest.mark.unit
    def test_valid_common_mistake_creation(self, sample_entity_data):
        """Test creating a valid CommonMistake."""
        data = sample_entity_data["common_mistake"]
        mistake = CommonMistake(**data)
        
        assert mistake.mistake_type == "premature_optimization"
        assert mistake.description == "Optimizing application code before identifying the actual bottleneck"
        assert mistake.prevention == "Always profile and measure before optimizing"
        assert mistake.warning_signs == "Assumptions about performance without data"
        assert mistake.domain == "backend"
        assert mistake.severity == "medium"

    @pytest.mark.unit
    def test_common_mistake_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError) as exc_info:
            CommonMistake()
        
        errors = exc_info.value.errors()
        required_fields = {"mistake_type", "description", "prevention"}
        error_fields = {error["loc"][0] for error in errors}
        
        assert required_fields.issubset(error_fields)

    @pytest.mark.unit
    def test_common_mistake_defaults(self):
        """Test default values for optional fields."""
        mistake = CommonMistake(
            mistake_type="test",
            description="test",
            prevention="test"
        )
        
        assert mistake.warning_signs == ""
        assert mistake.domain == ""
        assert mistake.severity == "medium"


class TestProblemContext:
    """Test ProblemContext entity type."""

    @pytest.mark.unit
    def test_valid_problem_context_creation(self):
        """Test creating a valid ProblemContext."""
        context = ProblemContext(
            problem_domain="backend",
            constraints="Limited budget and time",
            stakeholders="Product team, customers",
            urgency="high",
            environment="Production system with high load"
        )
        
        assert context.problem_domain == "backend"
        assert context.constraints == "Limited budget and time"
        assert context.stakeholders == "Product team, customers"
        assert context.urgency == "high"
        assert context.environment == "Production system with high load"

    @pytest.mark.unit
    def test_problem_context_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError) as exc_info:
            ProblemContext()
        
        errors = exc_info.value.errors()
        required_fields = {"problem_domain"}
        error_fields = {error["loc"][0] for error in errors}
        
        assert required_fields.issubset(error_fields)

    @pytest.mark.unit
    def test_problem_context_defaults(self):
        """Test default values for optional fields."""
        context = ProblemContext(problem_domain="test")
        
        assert context.constraints == ""
        assert context.stakeholders == ""
        assert context.urgency == "medium"
        assert context.environment == ""


class TestSuccessPattern:
    """Test SuccessPattern entity type."""

    @pytest.mark.unit
    def test_valid_success_pattern_creation(self):
        """Test creating a valid SuccessPattern."""
        pattern = SuccessPattern(
            pattern_name="Database Optimization Pattern",
            description="Systematic approach to database performance optimization",
            conditions="When database queries are the bottleneck",
            domain="backend",
            success_rate="85%",
            prerequisites="Database access and profiling tools"
        )
        
        assert pattern.pattern_name == "Database Optimization Pattern"
        assert pattern.description == "Systematic approach to database performance optimization"
        assert pattern.conditions == "When database queries are the bottleneck"
        assert pattern.domain == "backend"
        assert pattern.success_rate == "85%"
        assert pattern.prerequisites == "Database access and profiling tools"

    @pytest.mark.unit
    def test_success_pattern_required_fields(self):
        """Test that required fields are enforced."""
        with pytest.raises(ValidationError) as exc_info:
            SuccessPattern()
        
        errors = exc_info.value.errors()
        required_fields = {"pattern_name", "description", "conditions"}
        error_fields = {error["loc"][0] for error in errors}
        
        assert required_fields.issubset(error_fields)

    @pytest.mark.unit
    def test_success_pattern_defaults(self):
        """Test default values for optional fields."""
        pattern = SuccessPattern(
            pattern_name="test",
            description="test",
            conditions="test"
        )
        
        assert pattern.domain == ""
        assert pattern.success_rate == ""
        assert pattern.prerequisites == ""


class TestMemoryEntityTypes:
    """Test the MEMORY_ENTITY_TYPES dictionary."""

    @pytest.mark.unit
    def test_memory_entity_types_structure(self):
        """Test that MEMORY_ENTITY_TYPES contains all expected entity types."""
        expected_types = {
            "ProblemSolution",
            "LessonLearned", 
            "CommonMistake",
            "ProblemContext",
            "SuccessPattern"
        }
        
        assert set(MEMORY_ENTITY_TYPES.keys()) == expected_types
        
        # Verify all values are classes
        for name, entity_class in MEMORY_ENTITY_TYPES.items():
            assert hasattr(entity_class, "__name__")
            assert entity_class.__name__ == name

    @pytest.mark.unit
    def test_entity_types_instantiation(self, sample_entity_data):
        """Test that all entity types can be instantiated."""
        # Test ProblemSolution
        solution = MEMORY_ENTITY_TYPES["ProblemSolution"](**sample_entity_data["problem_solution"])
        assert isinstance(solution, ProblemSolution)
        
        # Test LessonLearned
        lesson = MEMORY_ENTITY_TYPES["LessonLearned"](**sample_entity_data["lesson_learned"])
        assert isinstance(lesson, LessonLearned)
        
        # Test CommonMistake
        mistake = MEMORY_ENTITY_TYPES["CommonMistake"](**sample_entity_data["common_mistake"])
        assert isinstance(mistake, CommonMistake)
