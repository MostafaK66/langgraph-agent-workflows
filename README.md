# LangGraph Agents Lab

A maintainable Python repository containing two LangGraph practice applications:

- a research assistant that can call a Tavily web-search tool;
- a five-paragraph essay writer that plans, researches, drafts, critiques, and revises.

The graph-independent node logic is fully testable offline. LangGraph, LangChain,
OpenAI, Tavily, and SQLite checkpoint integrations are loaded only by runtime commands.

## Security notice

The original repository tracked a `.env` file containing OpenAI and Tavily credentials.
That file has been removed and is now ignored. Because Git retains earlier versions,
revoke and replace both historical credentials before running this application. Never
commit the replacement values. Copy `.env.example` to `.env` for local use.

## Requirements and installation

- Python 3.11 or 3.12
- OpenAI and Tavily API credentials for live commands

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[agents]'
cp .env.example .env
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -e ".[agents]"
Copy-Item .env.example .env
```

Set `OPENAI_API_KEY` and `TAVILY_API_KEY` in `.env` or the process environment. Copy
`config.example.toml` to `config.toml` for non-secret settings. Relative checkpoint paths
are resolved from that file. The default `:memory:` database leaves no data on disk.

The original `gpt-3.5-turbo` default was changed to `gpt-4o-mini`, which the
[official OpenAI model documentation](https://developers.openai.com/api/docs/models/gpt-3.5-turbo)
identifies as its cheaper, more capable replacement. The model remains configurable.

## Usage

```bash
langgraph-agents --config config.toml research "What is LangGraph?"
langgraph-agents --config config.toml research --review-tools "What is LangGraph?"
langgraph-agents --config config.toml essay "The impact of urban trees"
```

For a multiline essay topic, omit the topic argument and finish standard input with
Ctrl+D on Linux/macOS or Ctrl+Z followed by Enter on Windows.

`--review-tools` preserves the original human-in-the-loop behavior at the useful safety
boundary: every proposed search is displayed and can be approved or skipped before any
network request runs. A bounded review loop prevents runaway tool execution.

## Architecture

- `messages.py` defines immutable messages and replacement-aware reduction.
- `research.py` implements model and tool nodes without importing LangChain.
- `essay.py` implements deterministic planning, research, revision, and stop logic.
- `graphs.py` assembles both LangGraph workflows and safely owns checkpointer lifetimes.
- `integrations.py` converts domain messages and validates OpenAI/Tavily responses.
- `runners.py` validates completed graph state.
- `cli.py` provides the composition root and catches domain errors.

The essay revision limit now means exactly that many drafts; the original comparison
generated one extra revision. Caller-owned content and message collections are never
mutated. Unknown or failed research tools return observations the model can recover from.

## Development

```bash
python -m pip install -e '.[dev]'
make quality
```

Quality checks run Ruff, strict mypy, deterministic pytest tests with at least 90%
branch-aware coverage, and bytecode compilation. Tests require no credentials, network,
model, external server, or persistent database.

## License and attribution

MIT licensed. This repository is based on the original
[ai_agents_langgraph](https://github.com/MostafaK66/ai_agents_langgraph) project by
MostafaK66. See `NOTICE` for attribution and the documented default-model migration.
