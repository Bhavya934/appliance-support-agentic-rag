import logging
from pathlib import PureWindowsPath
from typing import Literal
from langchain_groq import ChatGroq
from langchain_tavily import TavilySearch
from langgraph.graph import StateGraph, START, END
from app.core.config import get_settings
from app.rag.state import AgentState, RouteDecision, EvidenceGrade
from app.rag.vectorstore import retrieve

logger = logging.getLogger(__name__)
settings = get_settings()

_llms: dict[str, ChatGroq] = {}
_web_search = None

def llm(fast: bool = False):
    """The answer-writing model, or (fast=True) the smaller model used for routing, grading and rewriting.

    Each Groq model has its own free-tier quota, and evaluation/decision_test.py showed the small model
    makes the routing and grading decisions as accurately as the large one.
    """
    model = settings.groq_fast_model if fast else settings.groq_model
    if model not in _llms:
        if not settings.groq_api_key:
            raise RuntimeError("GROQ_API_KEY is missing")
        _llms[model] = ChatGroq(
            model=model,
            temperature=0,
            api_key=settings.groq_api_key,
            max_retries=6,  # free tier: back off on 429 rate limits instead of failing
        )
    return _llms[model]

def web_search_tool():
    global _web_search
    if _web_search is None:
        if not settings.tavily_api_key:
            raise RuntimeError("TAVILY_API_KEY is missing")
        _web_search = TavilySearch(
            tavily_api_key=settings.tavily_api_key,
            max_results=5,
            topic="general",
            include_answer=True,
            include_raw_content=False,
        )
    return _web_search

def add_trace(state: AgentState, message: str):
    return [*state.get("trace", []), message]

def doc_label(d) -> str:
    m = d.metadata
    if "brand" in m:
        return f"{m['brand']} {m['category'].replace('_', ' ')} {m['model']}, p.{int(m['page'])}"
    return PureWindowsPath(m.get("source", "unknown")).name

def route_prompt(question: str) -> str:
    return f"""
You route messages for a home-appliance support assistant backed by Bosch and Siemens user manuals
(washing machines, dryers, dishwashers, refrigerators, ovens, induction hobs).
Use kb for ANY question that mentions or concerns a home appliance, appliance brand or model: using,
installing, cleaning, maintaining, troubleshooting, error codes, safety, warranty, and also prices, availability,
recalls, spare parts, service costs, comparisons or news. (kb searches the manuals first and the web if needed.)
Use direct for messages that need no appliance information: greetings, thanks, goodbyes, compliments,
short acknowledgements ("ok", "sure"), and questions about the assistant itself ("what can you do?").
If a message mentions an appliance and you are unsure, use kb.
Question: {question}
Return valid JSON like {{"route":"kb"}}.
"""

def grade_kb_prompt(question: str, docs) -> str:
    context = "\n\n".join(f"Source: {doc_label(d)}\n{d.page_content}" for d in docs)
    return f"""
You grade evidence for a home-appliance support assistant.
Question: {question}
Evidence from the appliance manuals:\n{context}
Return good if the evidence is sufficient to answer confidently and specifically.
If the question names a brand or model, the evidence must be about that brand or model.
If it names only the appliance type (e.g. "my washing machine"), evidence from any manual for that type is fine,
even if it comes from several different models.
Otherwise return weak. JSON: {{"grade":"good"}} or {{"grade":"weak"}}.
"""

def route_question(state: AgentState):
    router = llm(fast=True).with_structured_output(RouteDecision, method="json_mode")
    decision = router.invoke(route_prompt(state["question"]))
    return {"source_used": decision.route, "trace": add_trace(state, f"Router → {decision.route.upper()}")}

def route_after_router(state: AgentState) -> Literal["retrieve_kb", "direct_answer"]:
    return "retrieve_kb" if state["source_used"] == "kb" else "direct_answer"

def retrieve_kb(state: AgentState):
    docs = retrieve(state["current_query"])
    how = f"reranked from {settings.rerank_candidates}" if settings.rerank_model else "vector search"
    return {"kb_docs": docs, "trace": add_trace(state, f"Manual retrieval → {len(docs)} chunks ({how})")}

def grade_kb(state: AgentState):
    grader = llm(fast=True).with_structured_output(EvidenceGrade, method="json_mode")
    grade = grader.invoke(grade_kb_prompt(state["question"], state["kb_docs"]))
    return {"kb_grade": grade.grade, "trace": add_trace(state, f"KB evidence grade → {grade.grade.upper()}")}

def after_kb(state: AgentState) -> Literal["generate_from_kb", "search_web"]:
    return "generate_from_kb" if state["kb_grade"] == "good" else "search_web"

def search_web(state: AgentState):
    result = web_search_tool().invoke({"query": state["current_query"]})
    lines, citations = [], []
    if isinstance(result, dict):
        if result.get("answer"):
            lines.append("Search answer: " + result["answer"])
        for item in result.get("results", []):
            title, url, content = item.get("title", ""), item.get("url", ""), item.get("content", "")
            lines.append(f"Title: {title}\nURL: {url}\nContent: {content}")
            citations.append({"title": title or url, "url": url, "type": "web"})
    else:
        lines.append(str(result))
    return {
        "web_results": "\n\n".join(lines),
        "citations": citations,
        "source_used": "web",
        "trace": add_trace(state, "Web fallback → Tavily search"),
    }

def grade_web(state: AgentState):
    grader = llm(fast=True).with_structured_output(EvidenceGrade, method="json_mode")
    grade = grader.invoke(f"""
Question: {state['question']}
Web evidence:\n{state['web_results']}
Return good if the evidence is sufficient and directly relevant; otherwise weak.
Return valid JSON like {{"grade":"good"}}.
""")
    return {"web_grade": grade.grade, "trace": add_trace(state, f"Web evidence grade → {grade.grade.upper()}")}

def after_web(state: AgentState) -> Literal["generate_from_web", "rewrite_query", "insufficient"]:
    if state["web_grade"] == "good":
        return "generate_from_web"
    if state["retry_count"] < settings.max_retries:
        return "rewrite_query"
    return "insufficient"

def rewrite_query(state: AgentState):
    rewritten = llm(fast=True).invoke(f"""
Rewrite this home-appliance support question for better retrieval from user manuals and web search.
Preserve intent, add useful technical keywords, do not answer, return only the query.
Question: {state['question']}
""").content.strip()
    return {
        "current_query": rewritten,
        "retry_count": state["retry_count"] + 1,
        "trace": add_trace(state, f"Query rewrite → {rewritten}"),
    }

def generate_from_kb(state: AgentState):
    context = "\n\n".join(f"[Source: {doc_label(d)}]\n{d.page_content}" for d in state["kb_docs"])
    answer = llm().invoke(f"""
You are a home-appliance support copilot. Answer ONLY from the manual excerpts below.
Be concise, actionable, and safe. If steps are present, present them clearly.
Do not invent settings, codes or specifications. Say which appliance manual the answer comes from.
For electrical, gas or water-connection repairs, tell the user to contact an authorised technician.
Question: {state['question']}\n\nManual excerpts:\n{context}
""").content
    citations = []
    seen = set()
    for d in state["kb_docs"]:
        label = doc_label(d)
        if label not in seen:
            seen.add(label)
            url = d.metadata.get("url", "")
            citations.append({"title": label, "url": f"{url}#page={int(d.metadata['page'])}" if url else "", "type": "private_kb"})
    return {"answer": answer, "source_used": "private_kb", "citations": citations, "trace": add_trace(state, "Answer generation → PRIVATE KB")}

def generate_from_web(state: AgentState):
    answer = llm().invoke(f"""
You are a home-appliance support copilot. The user manuals did not cover this question.
Answer ONLY from the web evidence below. Clearly say this is external web information that may not match the user's exact model,
and recommend checking the manufacturer's official support for safety-critical steps.
Question: {state['question']}\n\nWeb evidence:\n{state['web_results']}
""").content
    return {"answer": answer, "source_used": "web_search", "trace": add_trace(state, "Answer generation → WEB SEARCH")}

def direct_answer(state: AgentState):
    answer = llm(fast=True).invoke(f"""
You are a home-appliance support assistant. You answer questions from official Bosch and Siemens manuals for
washing machines, dryers, dishwashers, refrigerators, ovens and induction hobs, and search the web for things
manuals don't cover (recalls, prices, news). Respond briefly and naturally to: {state['question']}
""").content
    return {"answer": answer, "source_used": "direct", "trace": add_trace(state, "Direct response → no retrieval")}

def insufficient(state: AgentState):
    return {
        "answer": "I couldn't find enough reliable evidence in the appliance manuals or external search to answer confidently. Please share your appliance's brand and model number, or contact the manufacturer's customer service.",
        "source_used": "insufficient_evidence",
        "trace": add_trace(state, "Stopped → insufficient reliable evidence"),
    }

def build_graph():
    graph = StateGraph(AgentState)
    for name, fn in {
        "route_question": route_question,
        "retrieve_kb": retrieve_kb,
        "grade_kb": grade_kb,
        "search_web": search_web,
        "grade_web": grade_web,
        "rewrite_query": rewrite_query,
        "generate_from_kb": generate_from_kb,
        "generate_from_web": generate_from_web,
        "direct_answer": direct_answer,
        "insufficient": insufficient,
    }.items():
        graph.add_node(name, fn)

    graph.add_edge(START, "route_question")
    graph.add_conditional_edges("route_question", route_after_router, {
        "retrieve_kb": "retrieve_kb", "direct_answer": "direct_answer"
    })
    graph.add_edge("retrieve_kb", "grade_kb")
    graph.add_conditional_edges("grade_kb", after_kb, {
        "generate_from_kb": "generate_from_kb", "search_web": "search_web"
    })
    graph.add_edge("search_web", "grade_web")
    graph.add_conditional_edges("grade_web", after_web, {
        "generate_from_web": "generate_from_web", "rewrite_query": "rewrite_query", "insufficient": "insufficient"
    })
    graph.add_edge("rewrite_query", "retrieve_kb")
    graph.add_edge("generate_from_kb", END)
    graph.add_edge("generate_from_web", END)
    graph.add_edge("direct_answer", END)
    graph.add_edge("insufficient", END)
    return graph.compile()

agent_graph = build_graph()

def ask(question: str):
    initial: AgentState = {
        "question": question,
        "current_query": question,
        "kb_docs": [],
        "web_results": "",
        "kb_grade": "",
        "web_grade": "",
        "answer": "",
        "source_used": "",
        "retry_count": 0,
        "trace": [],
        "citations": [],
    }
    return agent_graph.invoke(initial)
