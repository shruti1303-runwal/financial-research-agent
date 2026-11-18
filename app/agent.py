"""
The research agent: Gemini + LangChain + RAG tool + MCP tools.

How a question flows:
  1. The LLM reads the question and decides which tool(s) to call.
  2. `search_financial_reports` (RAG) pulls relevant chunks from pgvector.
  3. MCP tools give live market data and exact arithmetic.
  4. The LLM may loop (call more tools) until it can answer, then writes the
     answer with [source, page] citations.

Try it in the terminal:  python -m app.agent "What was NovaTech's revenue growth in FY2025?"
"""
import asyncio
import json
import sys

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.tools import tool
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_mcp_adapters.client import MultiServerMCPClient

from app import config
from app.vectorstore import format_citation, retrieve

SYSTEM_PROMPT = """You are a careful financial research analyst.

Rules:
- For any question about a company's reports (revenue, profit, risks, strategy, guidance),
  FIRST call `search_financial_reports`. Answer ONLY from what it returns.
- Cite every fact from the reports inline like [annual_report.md, p.3].
- For live prices, returns or market metrics, use the market-data tools.
- For ANY arithmetic (growth, ratios, CAGR, differences), call a calculator tool. Never do math in your head.
- Break multi-part questions into steps and call tools as many times as needed.
- If the reports do not contain the answer, say so clearly. Never invent numbers.
- Finish with a short, well-structured answer (bullets are fine)."""


@tool(response_format="content_and_artifact")
def search_financial_reports(query: str) -> tuple[str, list[dict]]:
    """Search the ingested company financial reports (annual reports, 10-Ks, earnings
    releases). Input a focused search query, e.g. 'NovaTech FY2025 total revenue'.
    Returns the most relevant passages with their source and page."""
    hits = retrieve(query)
    if not hits:
        return "No relevant passages found.", []
    blocks, artifact = [], []
    for doc, dist in hits:
        cite = format_citation(doc)
        blocks.append(f"[{cite}]\n{doc.page_content}")
        artifact.append({"citation": cite, "source": doc.metadata.get("source"),
                         "page": doc.metadata.get("page"), "score": round(1 - float(dist), 4),
                         "text": doc.page_content})
    return "\n\n---\n\n".join(blocks), artifact


def mcp_client() -> MultiServerMCPClient:
    return MultiServerMCPClient({
        "finance": {"command": sys.executable, "args": [str(config.MCP_SERVER_PATH)],
                    "transport": "stdio"},
    })


async def build_agent(llm=None):
    """Create the agent. Loads tools from the MCP server + our local RAG tool."""
    mcp_tools = await mcp_client().get_tools()
    tools = [search_financial_reports, *mcp_tools]
    llm = llm or ChatGoogleGenerativeAI(model=config.CHAT_MODEL, temperature=0,
                                        api_key=config.GOOGLE_API_KEY or None)
    return create_agent(llm, tools, system_prompt=SYSTEM_PROMPT), [t.name for t in tools]


def _text(msg) -> str:
    c = msg.content
    if isinstance(c, str):
        return c
    return "".join(b.get("text", "") if isinstance(b, dict) else str(b) for b in c)


async def ask(agent, question: str) -> dict:
    """Run the agent and return answer + citations + the tool-call trace."""
    result = await agent.ainvoke({"messages": [{"role": "user", "content": question}]},
                                 {"recursion_limit": 25})
    msgs = result["messages"]
    steps, by_id, sources, contexts = [], {}, {}, []
    for m in msgs:
        if isinstance(m, AIMessage):
            for tc in m.tool_calls or []:
                step = {"tool": tc["name"], "args": tc["args"]}
                steps.append(step)
                by_id[tc["id"]] = step
        elif isinstance(m, ToolMessage):
            if m.name == "search_financial_reports" and m.artifact:
                for a in m.artifact:
                    sources.setdefault(a["citation"], a)
                    contexts.append(a["text"])
            elif m.tool_call_id in by_id:
                by_id[m.tool_call_id]["result"] = _text(m)[:500]
    return {"question": question, "answer": _text(msgs[-1]), "tool_calls": steps,
            "sources": list(sources.values()), "contexts": contexts}


async def _main(q: str):
    agent, names = await build_agent()
    print("Tools:", names)
    out = await ask(agent, q)
    print("\n=== TOOL CALLS ===")
    for s in out["tool_calls"]:
        print(f"- {s['tool']}({json.dumps(s['args'])})")
    print("\n=== ANSWER ===\n" + out["answer"])
    print("\n=== SOURCES ===")
    for s in out["sources"]:
        print(f"- {s['citation']} (similarity {s['score']})")


if __name__ == "__main__":
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252 (no ₹)
    asyncio.run(_main(" ".join(sys.argv[1:]) or
                      "What was NovaTech's revenue growth in FY2025 and what drove it?"))
