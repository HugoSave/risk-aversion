
# How to run

Install `uv` as package manager.

## Generation

`uv run scripts/risk_aversion_generator.py`

This will generate new json files in `./data`

## Evaluation

1. Set your provider API keys, `OPENAI_API_KEY`, `OPENROUTER_API_KEY` etc.
2. run the evaluations through `uv run inspect eval scripts/risk_aversion_eval.py --model openrouter/anthropic/claude-sonnet-4.5`
3. you can inspect the logs in detail with the Inspect AI plugin for VS code or through their viewer `uv run inspect view`

**Bias option**:
You can bias the models with a bias argument as follows:
```bash
uv run inspect eval scripts/risk_aversion_eval.py --model openrouter/anthropic/claude-sonnet-4.5 -T bias=risk_seeking
```
