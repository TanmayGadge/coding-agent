from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.stream.transformers import UpdatesTransformer
from langgraph.graph.message import add_messages

from langchain_core.messages import HumanMessage, BaseMessage, SystemMessage, AIMessage, ToolMessage
from langchain_openai import ChatOpenAI
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.tools import tool

from e2b_code_interpreter import Sandbox
from e2b.sandbox.commands.command_handle import CommandExitException

from dotenv import load_dotenv, find_dotenv, dotenv_values
from typing import Annotated, cast, Optional
from pydantic import BaseModel, Field, SecretStr
from os import getenv
from pathlib import Path

from output_schemas.test_results.test_result import TestResult, ActionableIssue

load_dotenv(find_dotenv())
SCRIPT_DIR = Path(__file__).parent

class TestingAgentState(BaseModel):
    code_path: str
    plan_path: str
    test_result: Optional[TestResult] = None
    code: Optional[str] = None
    feedback_to_coder: Optional[list[ActionableIssue]] = None
    messages: Annotated[list[BaseMessage], add_messages]
    
    failure_target: Optional[str] = None  # "SOURCE_CODE", "TEST_HARNESS", or "NONE"
    tester_retries: int = 0
    max_tester_retries: int = 3
    

@tool
def execute_code_in_sandbox(code: str):
    """Executes code in the cloud E2B sandbox and returns output for the tester"""
    
    env_values = {
        key: value for key, value in dotenv_values(find_dotenv()).items()
        if value is not None
    }
    with Sandbox.create(envs=env_values) as sandbox:
        # Pre-install dependencies
        sandbox.commands.run("pip install -q pytest pytest-asyncio httpx pydantic nest_asyncio")
        sandbox.files.write("runner.py", code)
        
        try:
            execution = sandbox.commands.run("python runner.py")
            output = []
            if execution.stdout:
                output.append(f"STDOUT:\n{execution.stdout}")
            if execution.stderr:
                output.append(f"STDERR:\n{execution.stderr}")
            return "\n".join(output) if output else "Execution completed with no output."
        except CommandExitException as e:
            # Catch non-zero exit codes so the LLM gets the error trace and self-corrects
            stdout = getattr(e, 'stdout', '') or ''
            stderr = getattr(e, 'stderr', e.stderr) or str(e)
            return f"EXECUTION FAILED (Exit Code {e.exit_code}):\nSTDOUT:\n{stdout}\nSTDERR:\n{stderr}"


model = ChatOpenAI(
    model="vibeconde-gpt-6-luna-medium",
    api_key= SecretStr(getenv("LITELLM_API_KEY") or ""),
    base_url=getenv("LITELLM_BASE_URL")
)
model_with_tools = model.bind_tools([execute_code_in_sandbox])

# small_model = ChatGoogleGenerativeAI(model="gemini-2.5-flash")

    
def code_reader(state: TestingAgentState):
    code_path = state.code_path
    global SCRIPT_DIR
    
    try:
        print(f"Reading code file from: {code_path}")
        
        abs_code_path = (SCRIPT_DIR / code_path).resolve()
        
        with open(abs_code_path, 'r', encoding="utf-8") as code_file:
            code = code_file.read()
        code_file.close()
        print(f"File read successfully!")
        
        return {"code": code}
    except FileNotFoundError:
        print(f"\n[code reader] Code file does not exist at the given location: {code_path}")
        return {"messages": [AIMessage(content="Could not read the file as it doesn't exist in the given location.")]}
    
def testing_agent_node(state: TestingAgentState):
    """Implement following software tests:
    1. Unit Testing
    2. Integration Testing
    3. System Testing"""
    global SCRIPT_DIR

    code = str(state.code)
    
    code_path_obj = Path(state.code_path)
    source_filename = code_path_obj.name            # e.g., "webhook_delivery.py" or "auth.py"
    module_stem = code_path_obj.stem                # e.g., "webhook_delivery" or "auth"
    test_filename = f"test_{module_stem}.py"
    
    abs_plan_path = (SCRIPT_DIR / state.plan_path).resolve()
    
    with open(abs_plan_path, 'r', encoding='utf-8') as plan_file:
        implementation_plan = plan_file.read()
    plan_file.close()
    
   
    user_prompt_text = (
        "--- IMPLEMENTATION PLAN ---\n"
        f"```{implementation_plan}```\n\n"
        "--- SOURCE CODE TO TEST ---\n"
        f"```python\n{code}\n```\n\n"
        "INSTRUCTIONS FOR YOUR TOOL CALL:\n"
        "Generate a SINGLE Python runner script to execute in the sandbox that performs the following sequence:\n"
        f"1. Writes the source code above into a file named `{source_filename}` using standard file I/O.\n"
        f"2. Writes your generated test cases into a file named `{test_filename}`.\n"
        f"3. Executes `pytest.main(['-v', '{test_filename}'])` and prints all execution output.\n"
    )
    
    if state.tester_retries > 0 and state.feedback_to_coder:
        issues_summary = "\n".join([f"- [{i.category}] {i.issue}" for i in state.feedback_to_coder])
        user_prompt_text += (
            f"\n\nPREVIOUS ATTEMPT FAILED (Retry {state.tester_retries}/{state.max_tester_retries}):\n"
            f"Your previous script caused test harness errors:\n{issues_summary}\n"
            "FIX THIS IN YOUR NEW TOOL CALL. Do NOT repeat the same runner/async error!"
        )
    
    system_msg = SystemMessage(
        content=(
            "You are an automated Software Testing Agent with access to a Python sandbox tool.\n\n"
            "NAMING CONVENTIONS FOR TESTS (CRITICAL):\n"
            "You MUST organize and name every test function in your test suite using explicit prefixes based on its scope:\n"
            "  - `test_unit_*`: For isolated class, function, or data-structure logic.\n"
            "  - `test_integration_*`: For interaction between components/modules or network/database calls with mocks.\n"
            "  - `test_system_*`: For end-to-end task flows against the implementation plan specs.\n\n"
            "EXECUTION WORKFLOW:\n"
            "1. Combine writing the source code file, writing the test file, and running pytest into ONE self-contained Python script.\n"
            "2. Call `execute_code_in_sandbox` to run your script.\n"
            "3. INSPECT SANDBOX OUTPUT BEFORE RESPONDING:\n"
            "   - If execution fails due to TEST HARNESS errors (e.g. `FileNotFoundError`, `SyntaxError`, `ImportError`, or event loop errors):\n"
            "     --> DO NOT return a final response yet!\n"
            "     --> Refactor your runner script to fix the missing file or syntax error.\n"
            "     --> Call `execute_code_in_sandbox` AGAIN to verify your fix.\n"
            "   - Only proceed to finalize your response when tests run cleanly (passing or legitimately failing source assertions).\n\n"
            "RUNNER SCRIPT REQUIREMENTS:\n"
            "- ALWAYS write the source file and test file to disk BEFORE importing them or running pytest.\n"
            "- Include `import nest_asyncio; nest_asyncio.apply()` at the top of your runner script.\n"
            "- For async test functions, use `@pytest.mark.asyncio` and `async def test_*()`.\n"
            "- DO NOT call `asyncio.run()` inside `@pytest.mark.asyncio` test functions."
        )
    )
    
    user_msg = HumanMessage(content=user_prompt_text)       
    response = model_with_tools.invoke([system_msg, user_msg] + state.messages)
    
    return {"messages": [response]}

def structure_node(state: TestingAgentState):
    structured_testing_model = model.with_structured_output(TestResult)
    
    system_msg = SystemMessage(
        content="You are an output structure agent.\n"
        "Analyze the pytest execution logs carefully and categorize test results into their respective levels:\n"
        "  - Place all tests named `test_unit_*` into `test_levels.unit`.\n"
        "  - Place all tests named `test_integration_*` into `test_levels.integration`.\n"
        "  - Place all tests named `test_system_*` into `test_levels.system`.\n\n"
        "DO NOT leave `integration` or `system` as null if matching test cases were executed. "
        "Populate summary counts, test case results, and metrics for each level executed."
    )
    
    response = cast(TestResult, structured_testing_model.invoke([system_msg] + state.messages))
    
    # Determine primary failure target
    failure_target = "NONE"
    if response.overall_status != "PASSED":
        has_test_harness_bug = any(
            issue.category == "TEST_HARNESS" for issue in response.actionable_issues_for_fixer
        )
        failure_target = "TEST_HARNESS" if has_test_harness_bug else "SOURCE_CODE"
        
    return {
        "test_result": response, 
        "feedback_to_coder": response.actionable_issues_for_fixer,
        "failure_target": failure_target,
        "tester_retries": state.tester_retries + (1 if failure_target == "TEST_HARNESS" else 0)
    }


def route_next_agent(state: TestingAgentState):
    """Route based on failure classification and retry counts."""
    target = state.failure_target
    retries = state.tester_retries
    max_retries = state.max_tester_retries

    if target == "TEST_HARNESS" and retries < max_retries:
        print(f"\n[Router] Test harness error detected. Retrying Testing Agent ({retries}/{max_retries})...")
        return "testing_agent"
    else:
        if target == "SOURCE_CODE":
            print("\n[Tester] Source code bugs detected. Summarizing findings for Coder...")
        else:
            print("\n[Tester] Testing complete. Summarizing results...")
        return "summary_agent"

def route_after_tester(state: TestingAgentState):
    last_msg = state.messages[-1]
    if isinstance(last_msg, AIMessage) and last_msg.tool_calls:
        return 'tools'
    else:
        return 'structure_agent'
    
def result_summariser(state: TestingAgentState):
    system_msg = SystemMessage(
        content="You are a software test result summariser.\n"
        "Your job is to take the test results given by the testing agent and summarise them so the developer can better understand it.\n"
        "You are to summarise them as follows:\n"
        "1. For all three tests (unit, integration and system), explain what went wrong or right and why.\n"
        "2. Explain the improvements suggested in by the tester agent in the provided JSON schema.\n"
        "Write your output in clean, clear natural language test format, NOT json.\n"
        "Keep the summary short, not more than 200 words."    
    )
    
    user_msg = HumanMessage(
        content="Summarise these test results.\n"
        "Test Results(separated by triple backticks):\n"
        f"```{state.test_result}```"
    )
    
    response = model.invoke([system_msg, user_msg])
    return {'messages': [response]}
        

tool_node = ToolNode(tools=[execute_code_in_sandbox])

graph = StateGraph(TestingAgentState)
graph.add_node('code_reader', code_reader)
graph.add_node('testing_agent', testing_agent_node)
graph.add_node('structure_agent', structure_node)
graph.add_node('tools', tool_node)
graph.add_node('summary_agent', result_summariser)

graph.add_edge(START, 'code_reader')
graph.add_edge('code_reader', 'testing_agent')
graph.add_conditional_edges('testing_agent', route_after_tester, {
    'tools': 'tools',
    'structure_agent': 'structure_agent'
})

graph.add_edge('tools', 'testing_agent')

graph.add_conditional_edges('structure_agent', route_next_agent, {
    'testing_agent': 'testing_agent', # Auto-fixes test harness in a loop
    'summary_agent': 'summary_agent'  # All passed or max retries reached
})

# graph.add_edge('structure_agent', 'summary_agent')
graph.add_edge('summary_agent', END)



if __name__ == "__main__":
    initial_input = TestingAgentState(
        code_path="../workspace/code_files/webhook_delivery.py",
        plan_path="../workspace/coding_plans/build-a-production-grade-asynchronous-webhook-deli.txt",
        test_result= None,
        code=None,
        messages=[],
        feedback_to_coder=None
    )
    
    app = graph.compile()
    
    stream = app.stream_events(initial_input, version='v3')
    
    for message in stream.messages:
        print(f"\n[{message.node}] ", end="")
        for delta in message.text:
            print(delta, end="", flush=True)
            