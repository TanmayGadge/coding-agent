A multi-agent system built using LangGraph.
Implements following agents:
1. Orchestrator: Controls the three agents responsible for the SDLC
2. Research Agent: Takes a task and builds an implementation plan by research tools & documentations from the web (Tavily search API).
3. Research Evaluator: An LLM-as-Judge eval node for the research agent. Assigns a score for the plan between 1-5 (based on pre-defined criteria). 
4. Coding Agent: Writes code for the given implementation plan. 
5. Testing Agent: Performs Unit, Integration & System tests on the code. Re-routes to the coding agent if tests fail.

Implemented message streaming via the stream_events API (v3) from LangGraph.
