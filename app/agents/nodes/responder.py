import logfire
from app.agents.state import AgentState
from app.gateway import chat_completion, extract_cache_status
from app.services.text_sanitize import strip_nul


TECHNICAL_PROMPT = """\
You are a Senior Technical Architect answering from company documentation.

Rules:
1. Use ONLY the TECHNICAL CONTEXT below. Every command, flag, name, value and claim in your
   answer must come from it. Do not add general knowledge, alternatives, best practices or
   background that the context does not state.
2. Start with the direct answer to the question, then give only the supporting steps or details the
   context provides. Quote exact commands and values from the context. Keep it short (a few
   sentences or a short list) and do not mention these rules.
3. If the context only covers part of the question, answer that part and add one sentence saying
   what is not covered in the documentation. If the context does not contain the answer, say so
   plainly instead of guessing.

TECHNICAL CONTEXT:
{context}

CONVERSATION HISTORY:
{history}

USER QUESTION:
"{question}"
"""


def build_technical_prompt(context: str, history: str, question: str) -> str:
    """The grounded RAG prompt. Kept separate so its rules are testable."""
    return TECHNICAL_PROMPT.format(context=context, history=history, question=question)


CONVERSATIONAL_PROMPT = """\
You are a friendly Enterprise AI Assistant for a documentation chatbot (Kubernetes, Redis, Databricks, Intel hardware, networking).
Reply to the user's latest message using the CONVERSATION HISTORY below.

Rules:
1. Answer only from the conversation history: greetings, thanks, and questions about what was said earlier.
2. Do not give technical explanations, comparisons, commands or recommendations from your own general knowledge.
   If the message needs that, say in one sentence that you answer from the documentation and ask the user to
   rephrase it as a specific technical question.
3. Keep it to a few sentences.

CONVERSATION HISTORY:
{history}

LATEST MESSAGE:
"{question}"
"""

TRUNCATION_NOTICE = "\n\n_(The answer was cut off at the length limit. Ask a narrower question for the rest.)_"


def build_conversational_prompt(history: str, question: str) -> str:
    return CONVERSATIONAL_PROMPT.format(history=history, question=question)


def mark_if_truncated(content: str, finish_reason) -> str:
    """The completion cap (MAX_COMPLETION_TOKENS) can stop an answer mid-sentence; say so."""
    return content + TRUNCATION_NOTICE if finish_reason == "length" else content


def generate_node(state: AgentState):
    """
    Synthesizes a response using both Documentation Context AND Conversation History.
    Uses the native Portkey client (not LangChain) so we can read the
    x-portkey-cache-status response header and surface Cache: Hit in the UI.
    """
    query = state["current_query"]

    history_str = ""
    for msg in state["messages"][:-1]:
        role = "User" if msg["role"] == "user" else "Assistant"
        history_str += f"{role}: {msg['content']}\n"

    user_msg = state["messages"][-1]["content"] if state["messages"] else ""

    if query == "CONVERSATIONAL":
        logfire.info("Generating conversational response using memory.")
        prompt = build_conversational_prompt(history_str, user_msg)
    else:
        logfire.info("Generating technical RAG response.")
        max_context_chars = 25000
        full_context = ""

        for doc in state["documents"]:
            if len(full_context) + len(doc) < max_context_chars:
                full_context += doc + "\n\n"
            else:
                logfire.warning("Context truncated to fit Groq TPM limits.")
                break

        prompt = build_technical_prompt(full_context, history_str, user_msg)

    with logfire.span("✍️ LLM Synthesis"):
        try:
            response = chat_completion(
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1
            )
            content = mark_if_truncated(
                strip_nul(response.choices[0].message.content or ""), response.choices[0].finish_reason
            )
            cache_status = extract_cache_status(response)
            is_cache_hit = cache_status == "HIT"

            if is_cache_hit:
                logfire.info("⚡ Gateway Cache Hit — response served from Portkey cache.")
                plan_update = state["plan"] + ["Cache: Hit ⚡"]
                status = "Cache hit — instant response."
            else:
                logfire.info("✅ Response synthesised via LLM.")
                plan_update = state["plan"]
                status = "Response generated."

            return {
                "final_answer": content,
                "status": status,
                "plan": plan_update,
                "messages": [{"role": "assistant", "content": content}]
            }

        except Exception as e:
            logfire.error(f"LLM Generation failed: {e}")
            raise e
