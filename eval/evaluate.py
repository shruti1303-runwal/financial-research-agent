"""
Evaluate the assistant on a custom question set.

Run:  python -m eval.evaluate                 # retrieval + end-to-end agent
      python -m eval.evaluate --retrieval-only

Metrics
-------
Retrieval (no LLM needed, cheap):
  * Hit@k   : did ANY expected source chunk appear in the top-k results?
  * MRR     : 1 / rank of the first relevant chunk (rewards ranking it high)
  * Recall@k: fraction of expected (file, section) pairs that were retrieved
Answer quality (LLM-as-judge + rules):
  * Faithfulness  : share of the answer's claims supported by retrieved context / tool output
  * Key-fact score: share of required facts (numbers) present in the answer
  * Citation rate : does the answer cite a source?
  * Refusal       : for unanswerable questions, did it say "not available" instead of inventing?
"""
import argparse
import asyncio
import json
import re
from pathlib import Path

from app import config
from app.vectorstore import retrieve

HERE = Path(__file__).parent


def retrieval_metrics(items, k):
    rows = []
    for it in items:
        exp = {tuple(e) for e in it["expected_sources"]}
        if not exp:
            continue
        hits = retrieve(it["question"], k=k)
        got = [(d.metadata.get("source"), d.metadata.get("page")) for d, _ in hits]
        rank = next((i + 1 for i, g in enumerate(got) if g in exp), None)
        rows.append({"id": it["id"], "hit": rank is not None,
                     "rr": 1 / rank if rank else 0.0,
                     "recall": len(exp & set(got)) / len(exp), "retrieved": got})
    n = len(rows)
    return {"k": k, "n": n,
            "hit_rate": round(sum(r["hit"] for r in rows) / n, 3),
            "mrr": round(sum(r["rr"] for r in rows) / n, 3),
            "recall": round(sum(r["recall"] for r in rows) / n, 3)}, rows


JUDGE_PROMPT = """You are grading an AI financial assistant for FAITHFULNESS.

CONTEXT (everything the assistant retrieved or computed with tools):
<<<
{context}
>>>

ANSWER:
<<<
{answer}
>>>

Break the answer into individual factual claims (numbers, names, statements).
For each claim decide if it is SUPPORTED by the context (directly stated, or a correct
calculation from stated numbers). Ignore generic phrasing that makes no factual claim.
Return ONLY JSON: {{"claims": [{{"claim": "...", "supported": true}}], "faithfulness": <supported/total, 1.0 if no claims>}}"""


def _norm(s: str) -> str:
    return s.replace(",", "").lower()


async def judge_faithfulness(llm, context: str, answer: str) -> float:
    msg = await llm.ainvoke(JUDGE_PROMPT.format(context=context[:20000], answer=answer))
    text = msg.content if isinstance(msg.content, str) else "".join(
        b.get("text", "") for b in msg.content if isinstance(b, dict))
    m = re.search(r"\{.*\}", text, re.S)
    try:
        return float(json.loads(m.group(0))["faithfulness"])
    except Exception:
        return float("nan")


async def answer_metrics(items):
    from langchain_google_genai import ChatGoogleGenerativeAI
    from app.agent import ask, build_agent

    agent, _ = await build_agent()
    judge = ChatGoogleGenerativeAI(model=config.JUDGE_MODEL, temperature=0,
                                   api_key=config.GOOGLE_API_KEY or None)
    rows = []
    for it in items:
        print(f"  {it['id']}: {it['question']}")
        out = await ask(agent, it["question"])
        ans = out["answer"]
        tool_ctx = "\n".join(f"{s['tool']}{s['args']} -> {s.get('result', '')}" for s in out["tool_calls"])
        context = "\n\n".join(out["contexts"]) + "\n\nTOOL RESULTS:\n" + tool_ctx
        row = {"id": it["id"], "question": it["question"], "answer": ans,
               "tools_used": [s["tool"] for s in out["tool_calls"]]}
        if it.get("expect_refusal"):
            row["refused_correctly"] = bool(re.search(
                r"not (available|found|contain|mention|provide|include)|no information|does not|doesn't|unable", ans, re.I))
        else:
            facts = it["key_facts"]
            row["key_fact_score"] = sum(_norm(f) in _norm(ans) for f in facts) / len(facts)
            row["cited"] = bool(re.search(r"\[[^\]]*\.(md|pdf|txt)", ans))
            row["faithfulness"] = await judge_faithfulness(judge, context, ans)
        rows.append(row)

    ans_rows = [r for r in rows if "key_fact_score" in r]
    ref_rows = [r for r in rows if "refused_correctly" in r]
    valid_f = [r["faithfulness"] for r in ans_rows if r["faithfulness"] == r["faithfulness"]]
    summary = {
        "n": len(rows),
        "faithfulness": round(sum(valid_f) / max(len(valid_f), 1), 3),
        "key_fact_accuracy": round(sum(r["key_fact_score"] for r in ans_rows) / len(ans_rows), 3),
        "citation_rate": round(sum(r["cited"] for r in ans_rows) / len(ans_rows), 3),
        "refusal_accuracy": round(sum(r["refused_correctly"] for r in ref_rows) / max(len(ref_rows), 1), 3),
        "avg_tool_calls": round(sum(len(r["tools_used"]) for r in rows) / len(rows), 2),
    }
    return summary, rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--k", type=int, default=config.TOP_K)
    ap.add_argument("--retrieval-only", action="store_true")
    a = ap.parse_args()
    items = json.loads((HERE / "questions.json").read_text(encoding="utf-8"))

    print("== Retrieval evaluation ==")
    r_sum, r_rows = retrieval_metrics(items, a.k)
    print(json.dumps(r_sum, indent=2))
    results = {"retrieval": r_sum, "retrieval_rows": r_rows}

    if not a.retrieval_only:
        print("\n== End-to-end agent evaluation ==")
        s, rows = asyncio.run(answer_metrics(items))
        print(json.dumps(s, indent=2))
        results.update({"answers": s, "answer_rows": rows})

    (HERE / "results.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")
    md = ["# Evaluation results", "", f"Questions: {len(items)} · top-k = {a.k}", "",
          "| Metric | Score |", "|---|---|",
          f"| Retrieval Hit@{a.k} | {r_sum['hit_rate']:.0%} |",
          f"| Retrieval MRR | {r_sum['mrr']:.2f} |",
          f"| Retrieval Recall@{a.k} | {r_sum['recall']:.0%} |"]
    if "answers" in results:
        s = results["answers"]
        md += [f"| Answer faithfulness (LLM judge) | {s['faithfulness']:.0%} |",
               f"| Key-fact accuracy | {s['key_fact_accuracy']:.0%} |",
               f"| Citation rate | {s['citation_rate']:.0%} |",
               f"| Correct refusal on unanswerable | {s['refusal_accuracy']:.0%} |",
               f"| Avg. tool calls per question | {s['avg_tool_calls']} |"]
    (HERE / "report.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print("\nSaved eval/results.json and eval/report.md")


if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding="utf-8")   # Windows consoles default to cp1252 (no ₹)
    main()
