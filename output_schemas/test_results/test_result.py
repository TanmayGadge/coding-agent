from enum import Enum
from typing import Optional
from pydantic import BaseModel, Field


class TestStatus(str, Enum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    ERROR = "ERROR"
    TIMEOUT = "TIMEOUT"
    SKIPPED = "SKIPPED"


# ==========================================
# 1. UNIT TESTING MODELS
# ==========================================

class TestCaseResult(BaseModel):
    test_name: str = Field(..., description="Name of the unit test case executed")
    status: TestStatus = Field(..., description="Status of this test case")
    duration_ms: Optional[float] = Field(None, description="Execution duration in milliseconds")
    mocks_verified: Optional[bool] = Field(None, description="Whether all mocks were verified")
    failure_type: Optional[str] = Field(None, description="Type of exception/assertion failure if failed")
    expected: Optional[str] = Field(None, description="Expected result or output")
    actual: Optional[str] = Field(None, description="Actual value or behavior observed")
    stack_trace: Optional[str] = Field(None, description="Traceback for failure diagnosis")
    unhandled_edge_case: Optional[str] = Field(None, description="Edge case that was not handled")


class ModuleUnitTest(BaseModel):
    target_module: str = Field(..., description="Path to the module under test")
    target_function: Optional[str] = Field(None, description="Target function or class name")
    test_cases: list[TestCaseResult] = Field(default_factory=list)


class UnitTestSummary(BaseModel):
    total_units_tested: int = Field(..., description="Total unit test cases run")
    passed: int = Field(..., description="Number of passed unit tests")
    failed: int = Field(..., description="Number of failed unit tests")
    coverage_percentage: float = Field(..., description="Overall unit statement coverage percentage")


class UnitTestMetrics(BaseModel):
    branch_coverage_pct: float = Field(..., description="Branch coverage percentage")
    untested_functions: list[str] = Field(default_factory=list, description="list of untested functions")


class UnitTestResult(BaseModel):
    summary: UnitTestSummary
    results: list[ModuleUnitTest] = Field(default_factory=list)
    metrics: UnitTestMetrics


# ==========================================
# 2. INTEGRATION TESTING MODELS
# ==========================================

class ContractCompliance(BaseModel):
    payload_schema_valid: bool = Field(..., description="Whether incoming request payload matches schema")
    response_schema_valid: bool = Field(..., description="Whether response payload matches schema")
    mismatched_fields: list[str] = Field(default_factory=list, description="Fields failing schema validation")


class DataIntegrity(BaseModel):
    transaction_rolled_back_on_error: bool = Field(..., description="Whether DB transactions roll back on failure")
    orphaned_records_detected: int = Field(..., description="Count of dangling or orphaned records found")


class ErrorCascade(BaseModel):
    upstream_handled_downstream_timeout: bool = Field(..., description="Whether upstream handles downstream timeouts")


class IntegrationTestCase(BaseModel):
    boundary: str = Field(..., description="System boundary tested (e.g. ServiceA -> ServiceB)")
    interface_type: str = Field(..., description="Interface protocol (e.g. REST_API, SQL, gRPC)")
    status: TestStatus = Field(..., description="Integration test outcome")
    contract_compliance: Optional[ContractCompliance] = None
    data_integrity: Optional[DataIntegrity] = None
    error_cascade: Optional[ErrorCascade] = None


class IntegrationTestSummary(BaseModel):
    total_interactions_tested: int
    passed: int
    failed: int


class IntegrationTestMetrics(BaseModel):
    api_contract_pass_rate: float = Field(..., description="Percentage of API schema contracts validated")
    data_consistency_status: str = Field(..., description="State of data consistency (e.g. CLEAN, CORRUPTED_ON_FAILURE)")
    avg_interface_latency_ms: float = Field(..., description="Average response latency across interfaces")


class IntegrationTestResult(BaseModel):
    summary: IntegrationTestSummary
    results: list[IntegrationTestCase] = Field(default_factory=list)
    metrics: IntegrationTestMetrics


# ==========================================
# 3. SYSTEM TESTING MODELS
# ==========================================

class RequirementTraceability(BaseModel):
    requirement_id: str = Field(..., description="Requirement ID from implementation plan (e.g. REQ-001)")
    description: str = Field(..., description="High-level description of the requirement")
    user_story: Optional[str] = Field(None, description="User story corresponding to requirement")
    status: TestStatus = Field(..., description="Status of requirement verification")
    end_to_end_flow: list[str] = Field(default_factory=list, description="Sequence of steps tested in workflow")
    acceptance_criteria_met: list[str] = Field(default_factory=list, description="Passed acceptance criteria")
    unmet_specifications: list[str] = Field(default_factory=list, description="Failed specification items")


class SlaLatencyCompliance(BaseModel):
    target_max_ms: float = Field(..., description="Target SLA max response time")
    p95_actual_ms: float = Field(..., description="Observed P95 response time")
    compliant: bool = Field(..., description="Whether SLA target was met")


class SystemTestSummary(BaseModel):
    requirements_verified: int
    requirements_passed: int
    requirements_failed: int
    specification_compliance_pct: float = Field(..., description="Percentage of overall requirements passed")


class SystemTestMetrics(BaseModel):
    e2e_workflow_completion_rate: float = Field(..., description="Completion rate of end-to-end user flows")
    sla_latency_compliance: SlaLatencyCompliance
    environment_status: str = Field(..., description="Deployment readiness (e.g. READY_FOR_DEPLOYMENT, BLOCKED)")


class SystemTestResult(BaseModel):
    summary: SystemTestSummary
    requirements_traceability: list[RequirementTraceability] = Field(default_factory=list)
    metrics: SystemTestMetrics


# ==========================================
# 4. TOP-LEVEL TEST RESULT ENVELOPE
# ==========================================

class ActionableIssue(BaseModel):
    level: str = Field(..., description="Testing level: 'unit', 'integration', or 'system'")
    file: Optional[str] = Field(None, description="Associated file path if applicable")
    category: str = Field(
        ..., 
        description="Target for fix: 'SOURCE_CODE' (bug in implementation) or 'TEST_HARNESS' (bug in test script/runner)"
    )
    requirement_id: Optional[str] = Field(None, description="Requirement ID if system test failure")
    issue: str = Field(..., description="Concise description of the bug or issue for the fixer agent")


class TestLevels(BaseModel):
    unit: Optional[UnitTestResult] = None
    integration: Optional[IntegrationTestResult] = None
    system: Optional[SystemTestResult] = None


class TestResult(BaseModel):
    overall_status: TestStatus = Field(..., description="Overall test suite result (PASSED if all pass)")
    execution_timestamp: str = Field(..., description="ISO 8601 timestamp of test run execution")
    test_levels: TestLevels = Field(..., description="Outputs for unit, integration, and system tests")
    next_action: str = Field(..., description="Routing signal for manager agent (e.g., ROUTE_TO_FIXER or PROCEED)")
    actionable_issues_for_fixer: list[ActionableIssue] = Field(
        default_factory=list, 
        description="Structured list of actionable bugs for the developer/fixer agent"
    )