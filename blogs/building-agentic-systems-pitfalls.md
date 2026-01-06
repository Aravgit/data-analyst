# The Hidden Pitfalls of Building Agentic AI Systems

2025 was called the year of agents. But until Claude Code shipped, truly agentic systems were sparse. I had built a multi-agent system earlier (https://thoughts.bitsnbytes.in/p/learnings-from-implementing-multi), but it was more orchestration than autonomy. Then it struck me: Claude Code, Codex—they're just while loops.

After exploring open source tools and using Claude Code and Codex, I built a data analysis agent. Upload a file, ask any question, and the agent figures out the columns it needs, runs the analysis, and returns a summary with charts. Worked great in development. Then it broke.

This article covers five pitfalls that don't show up in tutorials but will definitely show up in your debugging logs.

---


## 1. Memory That Never Dies

**The assumption**: Each agent turn is stateless. Resources clean up automatically.

**The reality**: Agents maintain state between tool calls—intermediate results, loaded resources, computed values. That state accumulates.

```python
def execute_tool(self, tool_name, args):
    result = run_tool(tool_name, args)
    self.resources[args['name']] = result  # Accumulates forever
    return result
```

Multiply by 100 concurrent users running multi-step chains. Memory climbs. Garbage collection can't help because references still exist. Eventually: OOM crashes.

**What happened**: Data files loaded for analysis stayed in RAM after each turn. Python's GC couldn't reclaim them because the session held references. With concurrent users, memory grew unbounded until containers crashed.

**The fix**: Explicit lifecycle management.

```python
def reset_after_turn(session):
    for key in list(session.resources.keys()):
        obj = session.resources.pop(key)
        del obj  # Break reference cycles
    session.resources = {}
    gc.collect()  # Force cleanup
```

Key insight: automatic garbage collection isn't enough for complex object graphs. You need explicit cleanup, explicit deletion, and explicit `gc.collect()`. Add TTL expiration and LRU eviction for multi-user deployments.

---

## 2. The Agent Loads Everything

**The assumption**: The model will make efficient choices about resource loading.

**The reality**: LLMs optimize for correctness, not efficiency. Given "load everything" vs "load only what's needed," they choose safety every time.

**What happened**: We provided three loading tools—sample (schema only), partial (specific columns), and full (everything). The agent consistently chose full loads even for "what's the average of column X?" Loading 50 columns when it needed 1.

**The fix**: It wasn't better tools—it was a better system prompt. Don't just provide tools; tell the agent *when* to use each:

```
LOADING STRATEGY (follow this order):
1. get_metadata() - Check what's available FIRST
2. load_partial(fields) - PRIMARY: Load ONLY what's needed
3. load_full() - RARE: Only when ALL fields required

Examples:
- "average sales by region" → load_partial(['region', 'sales'])
- "filter by date" → load_partial(['date', 'amount'])
```

The pattern: provide a clear hierarchy with concrete examples. Its still prompt engineering in 2026.

---

## 3. Context Overflow Mid-Turn

**The assumption**: Context windows are large enough.

**The reality**: In practice I found 100k–250k to be the practical range . Agent loops accumulate context fast. Tool calls, results, reasoning—it adds up. Hit the limit mid-turn and the user gets nothing.

A typical turn: user message (100 tokens) + system prompt (1000) + tool call (300) + tool result (2000) + another call (100) + another result (1500)...

After few turns, you're at 80% capacity. One unexpectedly large tool result (data analysis of 1000*10 rows and columns can explode the token count) and you hit the wall—mid-turn. The model can't complete. The user's question goes unanswered.

**What happened**: Our compaction logic triggered mid-turn when tokens exceeded the threshold. This interrupted the agent's reasoning. Users asked questions and got nothing back.

**The fix**: Compact *between* turns, never during. Also identify the parts to compact (messages, context, data), remove anything that does not help the agent in next turn. 

```python
def run_agent_turn(session, message):
    # Check BEFORE the turn, not during
    if session.total_tokens > (MAX_TOKENS * 0.80):
        compact_history(session)
    return execute_turn(session, message)
```

Check at the start, compact if needed, then proceed with guaranteed headroom. Don't punish the user for token limits. Track usage and proactively compact before hitting the ceiling(unless your kpi is token usage, and not outcomes reached).

---

## 4. Tool Schemas Are Your API Contract

**The assumption**: Tool descriptions are just documentation for the model.

**The reality**: Tool schemas are the contract between the model *and* your tool runner. The model uses them to decide what to call and how. Your runner uses them to validate and execute. Vague schemas = wrong calls = runtime errors on both sides.

```python
{
    "name": "process_data",
    "description": "Process the data",
    "parameters": {
        "data": {"type": "string"},
        "options": {"type": "object"}
    }
}
```

What data? What format? What options? The model guesses—and guesses wrong.

**The fix**: Strict, explicit schemas:

```python
{
    "name": "load_dataset_columns",
    "description": "Load specific columns from a dataset. Returns DataFrame.",
    "parameters": {
        "dataset_name": {
            "type": "string",
            "description": "Name from get_dataset_info()"
        },
        "columns": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Column names to load"
        },
        "df_name": {
            "type": "string",
            "description": "Variable name for the DataFrame in REPL"
        }
    },
    "required": ["dataset_name", "columns", "df_name"]
}
```

Rules: use explicit enums over free-form strings, describe relationships between tools ("use output from X"), document return values, mark required fields. The model can only be as precise as your schema allows.

---

## 5. Runaway Loops Will Burn Your Budget

**The assumption**: The agent will finish in a reasonable number of steps.

**The reality**: Without limits, agents can loop indefinitely. A confused model retries the same failing tool. A complex query spawns endless sub-tasks. Each iteration costs tokens, time, and money.

**What happened**: An edge case caused the agent to repeatedly call the same tool with slightly different parameters, expecting different results. 

**The fix**: Hard limits at multiple levels:

```
python
MAX_TOOL_CALLS_PER_TURN = 10
MAX_RETRIES_SAME_TOOL = 3

def run_agent_turn(session, message):
    tool_calls = 0
    tool_counts = {}

    while not done:
        response = model.generate(...)
        if response.has_tool_calls:
            for call in response.tool_calls:
                tool_counts[call.name] = tool_counts.get(call.name, 0) + 1
                if tool_counts[call.name] > MAX_RETRIES_SAME_TOOL:
                    return {"error": "Tool retry limit exceeded"}
            tool_calls += len(response.tool_calls)
            if tool_calls > MAX_TOOL_CALLS_PER_TURN:
                return {"error": "Tool call limit exceeded"}
```

Set sensible defaults: max tool calls per turn, max retries for the same tool, timeout per turn. When limits hit, fail gracefully with a clear message. The user can retry; 

---

## Why Not Use Agentic Libraries?

You might ask: why build from primitives instead of using LangChain, CrewAI, or similar frameworks?

Honestly, I chose not to. These libraries are still maturing, and the tools I admired—Claude Code, Codex—were built on raw foundations, not abstractions. I wanted to understand the while loop, not hide it.

That said, frameworks have their place. If you need to ship fast and your use case fits their patterns, use them. But if you're debugging a memory leak at 2:30 AM (Claude code limit reached :P), you'll want to know what's actually happening under the hood.

---

## Conclusion


The biggest pitfall is thinking an agent is just "an LLM in while loop with tools." It's not. It's a system with:

- **State management** (sessions, resources, context)
- **Resource constraints** (memory, tokens, time)
- **Failure modes** (partial, silent, cascading)
- **Concurrency concerns** (multiple users, multiple turns)

The while loop is the easy part. Everything around it—lifecycle management, resource efficiency, error handling, prompt engineering—that's where the work lives.

Build agents like you'd build any production system: explicit resource management, defensive error handling, clear contracts, and hard limits everywhere.

---

