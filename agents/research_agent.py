from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.graph.message import add_messages


from langchain_core.messages import BaseMessage, SystemMessage, AIMessage, HumanMessage
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_tavily import TavilySearch
from langchain_openai import ChatOpenAI


from dotenv import load_dotenv, find_dotenv
from typing import Annotated, cast
from pydantic import BaseModel, Field, SecretStr
from os import getenv
from pathlib import Path

load_dotenv(find_dotenv())
SCRIPT_DIR = Path(__file__).parent


class ResearchAgentState(BaseModel):
    messages: Annotated[list[BaseMessage], add_messages]
    task: str
    task_id: str
    coding_plan: str
    plan_path: str
    plan_score: int
    plan_feedback: str
    is_plan_valid: bool
    plan_retry_count: int = 0

search_tool = TavilySearch(
    max_results = 3,
    include_answers=True,
    topic="general",
    api_key=getenv("TAVILY_API_KEY")
)

tools = [search_tool]
tool_map = {tool.name: tool for tool in tools}

# model = ChatGoogleGenerativeAI(model='gemini-2.5-flash')
model = ChatOpenAI(
    model="vibeconde-gpt-6-luna-medium",
    api_key=SecretStr(getenv("LITELLM_API_KEY") or ""),
    base_url=getenv("LITELLM_BASE_URL")
)
model_with_tools = model.bind_tools(tools)    

eval_model = ChatOpenAI(
    model="vibeconde-gpt-6-luna-medium",
    api_key=SecretStr(getenv("LITELLM_API_KEY") or ""),
    base_url=getenv("LITELLM_BASE_URL")
)
# eval_model = ChatGoogleGenerativeAI(model='gemini-3.6-flash')

RESEARCH_ITERATION = 0

def research_agent_node(state: ResearchAgentState):
    global RESEARCH_ITERATION
    system_msg = SystemMessage(
        content="You are a Lead Software Architect research agent.\n"
        "Your job is to search for technical documentation, best practices, and design patterns from the web, "
        "then compile a conceptual, step-by-step architectural specification for the coding agent.\n\n"
        "STRICT OUTPUT RULES:\n"
        "1. DO NOT write any executable programming language code (Python, C++, etc.).\n"
        "2. DO NOT output any markdown code blocks (never use ```python or ```).\n"
        "3. Express all implementation steps strictly using structured bullet points, textual diagrams, or high-level pseudocode logic.\n"
        "4. Focus only on describing data structures, locking strategies, algorithm flow, and method signatures.\n"
        "The implementation plan should be extremely detailed (expected to be around 300-400 lines)"
    )
    
    if state.plan_feedback:
        user_prompt = f"""
        Consider the following feedback from the evaluator and make changes to the generated coding plan accordingly
        (separated by triple backticks):
        
        ```
        {state.plan_feedback}
        ```
        """
    else:
        user_prompt=""
        
    
    messages = [system_msg, user_prompt] + state.messages
    
    print(f'\n[Research Node]: RESEARCH_ITERATION = {state.plan_retry_count}')
    if state.plan_retry_count < 3:
        
        print(f"Refining the implementation plan based on the user's feedback")
        response = model_with_tools.invoke(messages)
        
        retry_count = state.plan_retry_count + 1
        # RESEARCH_ITERATION += 1
        
        return {'messages': [response], "plan_retry_count": retry_count}
    else:
        print("Max iterations hit.\nExiting research node...")
        return {}

def generate_filename(task: str, task_id: str = "") -> str:
    import re
    
    """Generates a safe filename slug from task_id or task content."""
    if task_id:
        base_name = task_id
    else:
        # Convert task to lowercase, replace non-alphanumeric chars with hyphens
        base_name = re.sub(r'[^a-zA-Z0-9]+', '-', task.lower()).strip('-')
        
        # Limit length to keep filename reasonable
        base_name = base_name[:50]
        
    return f"{base_name}.txt"

def structure_node(state: ResearchAgentState):
    global SCRIPT_DIR
    
    filename = generate_filename(task=state.task, task_id=state.task_id)
    
    output_dir = Path("../workspace/coding_plans")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    file_path = output_dir / filename
    abs_file_path = (SCRIPT_DIR / file_path).resolve()
    
    last_msg = state.messages[-1]
    coding_plan = last_msg.content
    
    try:
        with open(abs_file_path, 'w', encoding='utf-8') as file:
            file.write(str(coding_plan))
        file.close()
        
        print(f"\n\n[structure_node] Successfully saved coding plan to: {abs_file_path}")
    except FileNotFoundError:
        print(f"File not found at the given location: {abs_file_path}")
    
    return {'coding_plan': str(coding_plan), "plan_path": str(abs_file_path)}

def should_continue(state: ResearchAgentState):
    last_message = state.messages[-1]
    
    if isinstance(last_message, AIMessage) and last_message.tool_calls:
        return "tools"
    return 'structure_agent'


class PlanEvaluation(BaseModel):
    completeness_score: int = Field(description="Score from 1 to 5 for completeness.")
    edge_case_score: int = Field(description="Score from 1 to 5 for edge-case coverage.")
    clarity_score: int = Field(description="Score from 1 to 5 for step clarity.")
    overall_score: int = Field(
        description="Overall quality rating from 1 (unusable) to 5 (production-ready)."
    )
    is_acceptable: bool = Field(
        description="True if overall_score >= 4 and the plan can be safely passed to the coding agent; False otherwise."
    )
    feedback: str = Field(
        description="Actionable feedback specifying exactly what needs to be added or fixed if overall_score < 4."
    )


def evaluate_plan_node(state: ResearchAgentState):
    """"LLM-as-Judge node to check plan quality"""
    
    eval_prompt = f"""
    You are a Senior Software Quality Auditor. Evaluate the following coding plan generated by a research agent for a coding task.

    --- TASK ---
    {state.task}

    --- PROPOSED CODING PLAN ---
    {state.coding_plan}

    --- EVALUATION RUBRIC ---
    1. Completeness: Does the plan outline all function signatures, logic steps, parameter types, and return values needed to implement the solution?
    2. Edge-Case Coverage: Does the plan explicitly handle invalid types, boundary values (0, 1, 2), negative values, and unexpected inputs?
    3. Clarity: Are the steps sequentially ordered and completely unambiguous for a coding model to translate directly into code?

    --- SCORING SCALE ---
    - 1 (Unusable): Completely inaccurate, missing core requirements, or unsafe.
    - 2 (Poor): Missing critical steps or primary edge cases; unusable without major additions.
    - 3 (Acceptable with Gaps): Main algorithm is sound, but misses subtle edge cases or detailed validation.
    - 4 (Good): Clear, logically complete, and covers key edge cases; ready for implementation.
    - 5 (Production-Ready): Exceptional detail, covers all edge cases, performance considerations, and input validation.

    Provide your score, feedback, and set `is_acceptable` to True ONLY if overall_score is 4 or 5.
    """
    
    structured_eval_model = eval_model.with_structured_output(PlanEvaluation)
    
    current_retries = state.plan_retry_count + 1
    
    eval_result = cast(PlanEvaluation, structured_eval_model.invoke(eval_prompt))
    return {
        "plan_score": eval_result.overall_score,
        "plan_feedback": eval_result.feedback,
        "is_plan_valid": eval_result.is_acceptable,
        "plan_retry_count": current_retries
    }
    


graph = StateGraph(ResearchAgentState)
graph.add_node('research_agent', research_agent_node)
graph.add_node('structure_agent', structure_node)
graph.add_node('tools', ToolNode(tools))
graph.add_node('evaluation_agent', evaluate_plan_node)

graph.add_edge(START, 'research_agent')
graph.add_conditional_edges('research_agent', should_continue, ['tools', 'structure_agent'])
graph.add_edge('tools', 'research_agent')
graph.add_edge('structure_agent', 'evaluation_agent')
graph.add_edge('evaluation_agent', END)




if __name__ == "__main__":
    from langgraph.checkpoint.memory import MemorySaver
    from langchain_core.runnables import RunnableConfig
    
    configuration: RunnableConfig = {
        'configurable': {'thread_id': 'thread_1'}
    }
    
    memory = MemorySaver()
    
    app = graph.compile(
        checkpointer=memory
    )
    
    initial_input = ResearchAgentState(
        messages=[
            HumanMessage(
                content="Generate a code implementation plan for a python script that generates a fibbonacci sequence upto a given numbers. The sequence length is taken as a parameter."
            )
        ],
        task_id="task-fibbonacci",
        task="Generate a code implementation plan for a python script that generates a fibbonacci sequence upto a given numbers. The sequence length is taken as a parameter.",
        coding_plan="",
        plan_path="",
        plan_score=0,
        plan_feedback="",
        is_plan_valid=False,
        plan_retry_count=0,
    )
    
    events = app.stream(initial_input, stream_mode='updates', config=configuration)
    
    for event in events:
        for node_name, updates in event.items():
            if 'coding_plan' in updates:
                coding_plan = updates['coding_plan']
                print(f"Captured coding plan from {node_name}.")
                print(f"Coding plan: {coding_plan}")
                
                # file_path = "../workspace/coding-plans/plan.txt"
                # print(f"Wrtting coding plan to: {file_path}")
                        
                # with open(file_path, 'w', encoding='utf-8') as file:
                #     file.write(coding_plan)        
                # file.close()
                
                # print("Coding plan written successfully.")