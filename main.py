from langgraph.graph import StateGraph, START, END
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.memory import MemorySaver
from langgraph.stream.transformers import UpdatesTransformer

from langchain_core.messages import HumanMessage, BaseMessage, AIMessage
from langchain_core.runnables import RunnableConfig

from typing import TypedDict, Optional
from dotenv import load_dotenv, find_dotenv
from pathlib import Path

from agents.research_agent import graph as reasearch_graph, ResearchAgentState
from agents.coding_agent import graph as coding_graph, CodingAgentState
from agents.testing_agent import graph as testing_graph, TestingAgentState
from output_schemas.test_results.test_result import ActionableIssue
from token_tracker import TokenTracker


load_dotenv(find_dotenv())


class SharedMultiAgentState(TypedDict):
    task_id: str
    task: str
    plan_path: str      #Created by the Research Agent
    plan_score: int
    plan_feedback: str
    plan_retry_count:int
    coding_plan: str
    feedback_to_coder: Optional[list[ActionableIssue]]
    coder_retries: int
    
    tool_path: str      #Created by the Coding Agent
    test_results: dict   #Created by the Evaluation Agent
    
    


def run_research_agent(state: SharedMultiAgentState):
    """Orchestrator node that calls the research agent sub-graph."""
    
    subgraph_input = ResearchAgentState(
        task= state["task"],
        task_id=state["task_id"],
        messages=[HumanMessage(content=state["task"])],
        coding_plan="",
        plan_path="",
        plan_score=0,
        plan_feedback=state['plan_feedback'],
        is_plan_valid=False,
        plan_retry_count=0,
    )
    
    research_app = reasearch_graph.compile()
    
    result = research_app.invoke(subgraph_input) #Research agent subgraph
    
    #Use filesystem middleware to store the coding plan in a .txt file
    #File name shoud be dynamic and descriptive of the task
    #Store the filepath in the state
    return {
        "plan_path": result['plan_path'],
        "plan_score": int(result['plan_score']),
        "plan_feedback": result['plan_feedback'],
        # "coding_plan": result['coding_plan']
    }
    
def run_coding_agent(state: SharedMultiAgentState):
    plan_path = state['plan_path']
    
    print(f"[Orchestrator: coder] Reading implementation plan from {plan_path}...")
    with open(plan_path, 'r', encoding='utf-8') as file:
        coding_plan = file.read()
    file.close()
    print(f"Plan read successfully!")
    issues = state['feedback_to_coder']
    
    
    subgraph_input = CodingAgentState(
        coding_plan=coding_plan,
        code="",
        code_path="",
        code_feedback=issues if issues else None,
        messages=[]
    )
    
    coding_app = coding_graph.compile()
    result = coding_app.invoke(subgraph_input)
    
    
    return {"tool_path": result['code_path'], 'coder_retries': result['coder_retries']}

def run_testing_agent(state: SharedMultiAgentState):
    """Run testing agent"""
    
    
    subgraph_input = TestingAgentState(
        code_path=state['tool_path'],
        plan_path=state['plan_path'],
        test_result=None,
        code=None,
        messages=[],
        feedback_to_coder=None
    )
    
    testing_app = testing_graph.compile()
    result = testing_app.invoke(subgraph_input)
    
    return {"test_results": result['test_result'], "feedback_to_coder": result["feedback_to_coder"]}
    
    
def should_replan(state: SharedMultiAgentState):
    if state.get("plan_score", 0) > 3 or state['plan_retry_count'] > 3:
        return "coder"
    else:
        return 'researcher'

def should_recode(state: SharedMultiAgentState):
    tester_feedback = state['feedback_to_coder']
    
    if tester_feedback and state['coder_retries'] < 5:
        print(f"\n\n[Orchestrator] Code feedback detected. Calling coding agent...", flush=True)
        return 'coder'
    else:
        return END

orchestrator = StateGraph(SharedMultiAgentState)
orchestrator.add_node('researcher', run_research_agent)
orchestrator.add_node('coder', run_coding_agent)
orchestrator.add_node('tester', run_testing_agent)

orchestrator.add_edge(START, 'researcher')
orchestrator.add_conditional_edges('researcher', should_replan, {
    'researcher': 'researcher',
    'coder': 'coder'
})
orchestrator.add_edge('coder', 'tester')
orchestrator.add_conditional_edges('tester', should_recode, {
    'coder': 'coder',
    END: END
})

configuration: RunnableConfig = {
    "configurable": {'thread_id': "main_thread_1"}
}

memory = MemorySaver()

main_app = orchestrator.compile()

if __name__ == "__main__":
    relative_path = "./workspace/tasks/facebook_finder.txt"
    
    script_dir = Path(__file__).parent
    abs_path = (script_dir / relative_path).resolve()
    
    with open(abs_path, 'r', encoding='utf-8') as file:
        task = file.read()
    file.close()
    
    initial_main_app_input: SharedMultiAgentState = {
        "task_id": "",
        
        # "task": """
        #     Build a thread-safe, high-performance In-Memory Key-Value Store in Python with a configurable maximum capacity. The system must meet the following technical requirements:

        #     1. Eviction & Expiration: Implement Least Recently Used (LRU) eviction when capacity is reached, alongside optional per-key Time-To-Live (TTL) expiration in seconds.
        #     2. Thread Safety: Ensure all operations (get, put, delete, clear, size) are thread-safe using fine-grained or reentrant locking.
        #     3. Background Cleanup: Include a non-blocking background daemon thread that periodically sweeps and cleans up expired keys without causing lock contention during active read/write operations.
        #     4. Callbacks: Support an optional event callback system that triggers a user-defined function whenever a key is evicted (due to capacity) or expired (due to TTL).
        #     5. Edge Case Handling: Gracefully handle race conditions where an expired key is accessed immediately before background cleanup, zero/negative TTL values, and thread teardown/cleanup on cache destruction.""",
            
            "task": task,
            "plan_path": "",
            "plan_score": 0,
            "plan_feedback": "",
            "test_results": {},
            "tool_path": "",
            "coding_plan":"",
            "feedback_to_coder": None,
            "plan_retry_count": 0,
            'coder_retries': 0
    }

    
    #Migrate from stream API to stream_events API
    for namespace, mode, payload in main_app.stream(
        initial_main_app_input, 
        stream_mode=["messages", "updates"], 
        config=configuration,
        subgraphs=True  # <--- CRITICAL: Listens to child subgraphs
    ):
        if mode == 'messages':
           message_chunk, metadata = payload
           content = getattr(message_chunk, 'content', message_chunk)
           if content and isinstance(content, str):
               print(content.encode('utf-8', errors='ignore').decode('utf-8'), end="", flush=True)
        elif mode == 'updates':
            if isinstance(payload, dict):
                for node_name, state_updates in payload.items():
                    if isinstance(state_updates, dict):
                        if "plan_score" in state_updates:
                            score = state_updates["plan_score"]
                            print(f"\n\n[State Update] Node '{node_name}' assigned plan_score: {score}".encode('utf-8', errors='ignore').decode('utf-8'), flush=True)
                        if "plan_feedback" in state_updates:
                            feedback = state_updates["plan_feedback"]
                            print(f"[State Update] Feedback: {feedback}\n".encode('utf-8', errors='ignore').decode('utf-8'), flush=True)

                        if "coding_plan" in state_updates:
                            print(f"\n[State Update] Node '{node_name}' created/updated the coding plan.".encode('utf-8', errors='ignore').decode('utf-8'), flush=True)
                    else:
                        print(f"State Updates is not an iterable.", f"state_updates: {state_updates}".encode('utf-8', errors='ignore').decode('utf-8'), sep="\n")
                        print(f"Node name: {node_name}".encode('utf-8', errors='ignore').decode('utf-8'))
                        print(f"payload.items():\n{payload.items()}".encode('utf-8', errors='ignore').decode('utf-8'))

    
    # tracker = TokenTracker()
    
    
    # with tracker.live():
    #     stream = main_app.stream_events(initial_main_app_input, version='v3')
    #     for message in stream.messages:
    #         print(f"[{message.node}] ", end="")
    #         for delta in message.text:
    #             print(delta, flush=True, end="")
       
    
    png_bytes = main_app.get_graph(xray=True).draw_mermaid_png()
    with open("full_system_graph.png", "wb") as f:
        f.write(png_bytes)

    print("Saved complete graph with sub-agents to full_system_graph.png")
            
