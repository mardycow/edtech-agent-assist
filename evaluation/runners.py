import uuid
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
from agent_workflow import app

def agent_planning(queries: dict) -> dict[str, str]:
    """
    Run the full agent workflow for a given query
    """
    user_query = queries["text"]
    thread_id = str(uuid.uuid4())
    config = {"configurable": {"thread_id": thread_id, "is_test": True}}

    final_state = app.invoke({"user_query": user_query}, config=config)

    used_tools = []
    for msg in final_state.get("messages", []):
        if hasattr(msg, 'tool_calls') and msg.tool_calls:
            for tc in msg.tool_calls:
                used_tools.append(tc.get('name', 'unknown'))
    
    return {
        "final_answer": final_state["final_answer"],
        "messages": final_state["messages"],
        "tools": used_tools 
    }
