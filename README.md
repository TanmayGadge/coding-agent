A multi-agent system built using LangGraph.
Implements following agents:
1. Orchestrator: Controls the three agents responsible for the SDLC
2. Research Agent: Takes a task and builds an implementation plan by research tools & documentations from the web (Tavily search API).
3. Research Evaluator: An LLM-as-Judge eval node for the research agent. Assigns a score for the plan between 1-5 (based on pre-defined criteria).
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
      
4. Coding Agent: Writes code for the given implementation plan. 
5. Testing Agent: Performs Unit, Integration & System tests on the code. Re-routes to the coding agent if tests fail.

Under Development:
- Token Tracker: Tracks token usage in real time.

Implemented message streaming via the stream_events API (v3) from LangGraph.
