"""
Custom entity types for AI Agent Memory System.
These extend the base Graphiti MCP server with problem-solving specific entities.
"""

from pydantic import BaseModel, Field


class ProblemSolution(BaseModel):
    """A ProblemSolution represents a successful approach to solving a specific type of problem.
    
    Instructions for identifying and extracting problem solutions:
    1. Look for successful resolution of technical or business problems
    2. Identify the specific approach or methodology used
    3. Extract the domain or context where the solution was applied
    4. Capture the effectiveness and any metrics of success
    5. Include any tools, technologies, or frameworks used
    6. Note any prerequisites or conditions for the solution to work
    7. Preserve the original problem context that led to this solution
    8. Include any variations or adaptations of the solution
    """
    
    problem_type: str = Field(
        ...,
        description="The category/type of problem this solution addresses (e.g., 'performance_optimization', 'debugging', 'architecture_design')"
    )
    domain: str = Field(
        ..., 
        description="The domain or field where this solution applies (e.g., 'backend_api', 'frontend_ui', 'database', 'devops')"
    )
    approach: str = Field(
        ...,
        description="Brief description of the solution approach and methodology used"
    )
    effectiveness: str = Field(
        ...,
        description="How effective this solution was (high/medium/low) with supporting evidence"
    )
    tools_used: str = Field(
        default="",
        description="Tools, technologies, or frameworks used in this solution"
    )
    complexity: str = Field(
        default="moderate",
        description="Complexity level of implementing this solution (simple/moderate/complex/expert)"
    )


class LessonLearned(BaseModel):
    """A LessonLearned captures important insights from problem-solving experiences.
    
    Instructions for identifying and extracting lessons learned:
    1. Look for explicit statements of learning or insight
    2. Identify patterns that emerged from the experience
    3. Extract actionable knowledge that can be applied elsewhere
    4. Capture the context that led to this learning
    5. Include any conditions or limitations for applying this lesson
    6. Note the level of confidence in this lesson
    7. Preserve the original experience that generated the insight
    8. Include any supporting evidence or examples
    """
    
    context: str = Field(
        ...,
        description="The context or situation where this lesson was learned"
    )
    insight: str = Field(
        ...,
        description="The key insight or lesson learned from the experience"
    )
    applicability: str = Field(
        ...,
        description="Where and when this lesson can be applied (scope and conditions)"
    )
    confidence_level: str = Field(
        default="medium",
        description="Confidence level in this lesson (low/medium/high) based on evidence"
    )
    domain: str = Field(
        default="",
        description="The domain where this lesson applies"
    )


class CommonMistake(BaseModel):
    """A CommonMistake represents a frequently made error and how to avoid it.
    
    Instructions for identifying and extracting common mistakes:
    1. Look for errors, failures, or suboptimal approaches
    2. Identify patterns of mistakes that occur repeatedly
    3. Extract the root causes behind the mistakes
    4. Capture the impact or consequences of the mistake
    5. Include prevention strategies and warning signs
    6. Note the context where this mistake commonly occurs
    7. Preserve examples of when this mistake was made
    8. Include any tools or practices that help prevent the mistake
    """
    
    mistake_type: str = Field(
        ...,
        description="The category of mistake (e.g., 'logic_error', 'configuration_error', 'design_flaw')"
    )
    description: str = Field(
        ...,
        description="Description of the mistake and its consequences"
    )
    prevention: str = Field(
        ...,
        description="How to prevent or avoid this mistake"
    )
    warning_signs: str = Field(
        default="",
        description="Early indicators that this mistake might be occurring"
    )
    domain: str = Field(
        default="",
        description="The domain where this mistake commonly occurs"
    )
    severity: str = Field(
        default="medium",
        description="Severity of this mistake (low/medium/high/critical)"
    )


class ProblemContext(BaseModel):
    """A ProblemContext captures the environmental and situational factors surrounding a problem.
    
    Instructions for identifying and extracting problem context:
    1. Look for environmental factors that influenced the problem
    2. Identify constraints, requirements, and limitations
    3. Extract stakeholder information and business context
    4. Capture technical environment and system state
    5. Include timeline and urgency factors
    6. Note any external dependencies or influences
    7. Preserve the original problem statement and goals
    8. Include any assumptions or hypotheses made
    """
    
    problem_domain: str = Field(
        ...,
        description="The domain where the problem occurred"
    )
    constraints: str = Field(
        default="",
        description="Constraints and limitations that affected problem-solving"
    )
    stakeholders: str = Field(
        default="",
        description="Key stakeholders and their interests"
    )
    urgency: str = Field(
        default="medium",
        description="Urgency level of the problem (low/medium/high/critical)"
    )
    environment: str = Field(
        default="",
        description="Technical or business environment context"
    )


class SuccessPattern(BaseModel):
    """A SuccessPattern represents a repeatable approach that consistently leads to good outcomes.
    
    Instructions for identifying and extracting success patterns:
    1. Look for approaches that have worked well multiple times
    2. Identify the key elements that make the pattern successful
    3. Extract the conditions under which this pattern works best
    4. Capture any variations or adaptations of the pattern
    5. Include metrics or evidence of success
    6. Note any prerequisites or setup required
    7. Preserve examples of successful applications
    8. Include any limitations or failure modes
    """
    
    pattern_name: str = Field(
        ...,
        description="Name or identifier for this success pattern"
    )
    description: str = Field(
        ...,
        description="Description of the pattern and how it works"
    )
    conditions: str = Field(
        ...,
        description="Conditions under which this pattern is most effective"
    )
    domain: str = Field(
        default="",
        description="Domain where this pattern applies"
    )
    success_rate: str = Field(
        default="",
        description="Success rate or effectiveness metrics if available"
    )
    prerequisites: str = Field(
        default="",
        description="Prerequisites or setup required for this pattern"
    )


# Memory-specific entity types dictionary
MEMORY_ENTITY_TYPES: dict[str, BaseModel] = {
    'ProblemSolution': ProblemSolution,  # type: ignore
    'LessonLearned': LessonLearned,  # type: ignore
    'CommonMistake': CommonMistake,  # type: ignore
    'ProblemContext': ProblemContext,  # type: ignore
    'SuccessPattern': SuccessPattern,  # type: ignore
}
