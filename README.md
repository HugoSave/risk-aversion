
# How to run
1. Install `uv` as package manager. Run `uv sync`.
2. Set your provider API keys, `OPENAI_API_KEY`, `OPENROUTER_API_KEY` etc.
3. run the evaluations through `uv run inspect eval scripts/risk_aversion_eval.py --model openrouter/anthropic/claude-sonnet-4.5`
4. you can inspect the logs in detail with the Inspect AI plugin for VS code or through their viewer `uv run inspect view`

**Bias option**:
You can bias the models with a bias argument as follows:
```bash
uv run inspect eval scripts/risk_aversion_eval.py --model openrouter/anthropic/claude-sonnet-4.5 -T bias=risk_seeking
```
