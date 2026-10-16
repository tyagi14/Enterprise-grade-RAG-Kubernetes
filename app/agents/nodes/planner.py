from app.agents.state import AgentState
from app.gateway import get_langchain_llm
import logfire
from app.services.text_sanitize import strip_nul

# Portkey-backed LLM: fallback + cache + retry — same .invoke() interface as ChatGroq
llm = get_langchain_llm(feature="planner")

PLANNER_PROMPT = """
    You are an intelligent Assistant Planner. 
    Analyze the conversation history and the latest user message.
    
    CONVERSATION HISTORY:
    {history}
    
    LATEST MESSAGE:
    "{user_message}"
    
    Task:
    1. If the latest message is a greeting (hi, hello) or a question that can be answered using ONLY the conversation history above (e.g., "what is my name"), respond with 'CONVERSATIONAL'.
    2. If it is anything else that is NOT a technical question — a statement about the user (e.g., "my name is Sam", "I work on the platform team"), thanks, small talk, or a short reply like "ok" — respond with 'CONVERSATIONAL'.
    3. If it is a technical question about IT or software — Kubernetes, Redis, Databricks, cloud and data platforms, CLIs and commands, Intel hardware, networking, configuration, deployment, monitoring — that could be answered from documentation, output a refined search query. A question about a specific tool, command or product is technical even if that tool is not named in this list. A comparison or recommendation between technologies ("which would you prefer", "X or Y for ...", "which is better") is also technical: search for what the documentation says about it.
    
    You must always answer. Output ONLY 'CONVERSATIONAL' or the search query.
    """


def build_planner_prompt(history: str, user_message: str) -> str:
    return PLANNER_PROMPT.format(history=history, user_message=user_message)


def planner_node(state: AgentState):
    """
    The Planner determines if a search is needed based on the ENTIRE conversation.
    """
    # Get the conversation history (excluding the latest message)
    history = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history += f"{role}: {msg['content']}\n"
    
    user_message = state["messages"][-1]["content"] if state["messages"] else ""
    
    prompt = build_planner_prompt(history, user_message)
    
    with logfire.span("🧠 Planner Decision"):
        # strip_nul: the model sometimes ends its answer with a NUL, which Postgres checkpointing rejects.
        decision = strip_nul(llm.invoke(prompt).content).strip()
        planner_fallback = not decision
        if planner_fallback:
            # Empty completion (Qdrant/Vertex reject an empty query) — search on the raw message.
            logfire.warning("Planner returned an empty decision — falling back to the user message.")
            decision = user_message
        logfire.info(f"Intent identified: {decision}")
    
    if decision == "CONVERSATIONAL":
        return {
            "current_query": "CONVERSATIONAL",
            "status": "Handling conversationally (using memory)...",
            "plan": ["Intent: Conversational/Memory", "Retrieval: Skipped"]
        }
    
    # A fallback turn was not actually classified as technical, so it gets a different marker:
    # the semantic cache only stores plans containing exactly "Intent: Technical".
    intent = "Intent: Technical (planner fallback)" if planner_fallback else "Intent: Technical"
    return {
        "current_query": decision,
        "status": f"Technical research needed. Searching for: {decision}",
        "plan": [intent, f"Search Term: {decision}"]
    }
