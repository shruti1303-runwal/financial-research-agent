"""
MCP server exposing market-data and calculator tools.

MCP (Model Context Protocol) is an open standard for giving LLMs tools.
Any MCP client (our LangChain agent, Claude Desktop, Cursor...) can launch this
server and call its tools, so the tools are written once and reused everywhere.

Run standalone to test:   mcp dev app/mcp_server.py       (opens the MCP Inspector)
The agent launches it automatically over stdio.
"""
import ast
import math
import operator

# Import yfinance (and with it numpy/pandas) at startup, not inside the tools.
# On Windows, loading numpy's compiled DLLs on a worker thread while the stdio
# transport is reading stdin can deadlock, so the first market-data call hangs forever.
import yfinance as yf
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("finance-tools")


# ----------------------------------------------------------------- market data
@mcp.tool()
def get_stock_quote(ticker: str) -> dict:
    """Get the latest stock price, previous close, day change and market cap.
    Use Yahoo Finance tickers, e.g. 'AAPL', 'MSFT', 'INFY.NS' (NSE), 'TCS.NS', 'RELIANCE.NS'."""
    try:
        t = yf.Ticker(ticker)
        fi = t.fast_info
        last, prev = float(fi["last_price"]), float(fi["previous_close"])
        return {"ticker": ticker.upper(), "price": round(last, 2),
                "previous_close": round(prev, 2),
                "day_change_pct": round((last / prev - 1) * 100, 2),
                "market_cap": fi.get("market_cap"), "currency": fi.get("currency")}
    except Exception as e:  # network errors, unknown ticker
        return {"error": f"Could not fetch quote for {ticker}: {e}"}


@mcp.tool()
def get_price_history(ticker: str, period: str = "1y") -> dict:
    """Summarise price performance over a period: '1mo', '3mo', '6mo', '1y', '2y', '5y', 'ytd'.
    Returns start/end price, total return %, high, low and annualised volatility %."""
    try:
        h = yf.Ticker(ticker).history(period=period)["Close"].dropna()
        if h.empty:
            return {"error": f"No price history for {ticker}"}
        rets = h.pct_change().dropna()
        return {"ticker": ticker.upper(), "period": period,
                "start_date": str(h.index[0].date()), "end_date": str(h.index[-1].date()),
                "start_price": round(float(h.iloc[0]), 2), "end_price": round(float(h.iloc[-1]), 2),
                "total_return_pct": round((h.iloc[-1] / h.iloc[0] - 1) * 100, 2),
                "high": round(float(h.max()), 2), "low": round(float(h.min()), 2),
                "annualised_volatility_pct": round(float(rets.std() * math.sqrt(252) * 100), 2)}
    except Exception as e:
        return {"error": f"Could not fetch history for {ticker}: {e}"}


@mcp.tool()
def get_key_metrics(ticker: str) -> dict:
    """Valuation and profitability metrics: P/E, EPS, profit margin, ROE, revenue, debt/equity."""
    try:
        info = yf.Ticker(ticker).info
        keys = {"trailingPE": "pe_ratio", "trailingEps": "eps", "profitMargins": "profit_margin",
                "returnOnEquity": "roe", "totalRevenue": "revenue", "debtToEquity": "debt_to_equity",
                "dividendYield": "dividend_yield", "sector": "sector", "longName": "name"}
        return {"ticker": ticker.upper(), **{v: info.get(k) for k, v in keys.items()}}
    except Exception as e:
        return {"error": f"Could not fetch metrics for {ticker}: {e}"}


# ----------------------------------------------------------------- calculators
_OPS = {ast.Add: operator.add, ast.Sub: operator.sub, ast.Mult: operator.mul,
        ast.Div: operator.truediv, ast.Pow: operator.pow, ast.Mod: operator.mod,
        ast.USub: operator.neg, ast.UAdd: operator.pos}
_FUNCS = {"sqrt": math.sqrt, "log": math.log, "abs": abs, "round": round,
          "min": min, "max": max}


def _safe_eval(node):
    """Evaluate arithmetic WITHOUT Python's eval() (which could run any code)."""
    if isinstance(node, ast.Expression):
        return _safe_eval(node.body)
    if isinstance(node, ast.Constant) and isinstance(node.value, (int, float)):
        return node.value
    if isinstance(node, ast.BinOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.left), _safe_eval(node.right))
    if isinstance(node, ast.UnaryOp) and type(node.op) in _OPS:
        return _OPS[type(node.op)](_safe_eval(node.operand))
    if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id in _FUNCS:
        return _FUNCS[node.func.id](*[_safe_eval(a) for a in node.args])
    raise ValueError("Unsupported expression")


@mcp.tool()
def calculate(expression: str) -> dict:
    """Evaluate an arithmetic expression, e.g. '(5120 - 4410) / 4410 * 100'.
    Supports + - * / ** %, parentheses, sqrt, log, abs, round, min, max.
    ALWAYS use this for arithmetic instead of computing in your head."""
    try:
        result = _safe_eval(ast.parse(expression.replace(",", ""), mode="eval"))
        return {"expression": expression, "result": round(float(result), 6)}
    except Exception as e:
        return {"error": f"Cannot evaluate '{expression}': {e}"}


@mcp.tool()
def growth_rate(old_value: float, new_value: float) -> dict:
    """Percentage change from old_value to new_value (e.g. year-over-year revenue growth)."""
    if old_value == 0:
        return {"error": "old_value cannot be 0"}
    return {"growth_pct": round((new_value / old_value - 1) * 100, 4)}


@mcp.tool()
def cagr(start_value: float, end_value: float, years: float) -> dict:
    """Compound annual growth rate between two values over a number of years."""
    if start_value <= 0 or years <= 0:
        return {"error": "start_value and years must be positive"}
    return {"cagr_pct": round(((end_value / start_value) ** (1 / years) - 1) * 100, 4)}


@mcp.tool()
def financial_ratio(numerator: float, denominator: float, ratio_name: str = "ratio") -> dict:
    """Compute any ratio (e.g. net margin = net_income / revenue, debt/equity, current ratio).
    Returns both the raw ratio and the percentage."""
    if denominator == 0:
        return {"error": "denominator cannot be 0"}
    r = numerator / denominator
    return {"ratio_name": ratio_name, "value": round(r, 6), "as_percent": round(r * 100, 4)}


if __name__ == "__main__":
    mcp.run()   # stdio transport by default
