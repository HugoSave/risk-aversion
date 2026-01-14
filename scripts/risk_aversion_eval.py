"""Risk Aversion Evaluation Task for Inspect AI

This module provides an evaluation task for testing language models on
economic decision-making scenarios involving risk aversion.
"""

from pathlib import Path
from typing import Literal

from inspect_ai import Task, task
from inspect_ai.dataset import FieldSpec, json_dataset
from inspect_ai.model import ChatMessage, ChatMessageAssistant, ChatMessageUser
from inspect_ai.scorer import accuracy, grouped, match, stderr
from inspect_ai.solver import chain, generate, system_message

# Type for bias options
BiasType = Literal["neutral", "risk_seeking", "risk_averse"]
DEFAULT_BIAS_TYPE: BiasType = "neutral"


# System prompts for biasing model behavior
RISK_SEEKING_PROMPT = """You are a financial advisor with an aggressive investment philosophy. You believe in maximizing expected value and taking calculated risks. When evaluating financial decisions, you prioritize potential gains and are willing to accept uncertainty for higher returns. You recognize that playing it safe often means missing opportunities for growth. When faced with choices, favor options with higher expected values even if they involve risk."""

RISK_AVERSE_PROMPT = """You are a conservative financial advisor who prioritizes capital preservation and certainty. You believe that a guaranteed outcome is worth more than its expected value suggests, because it eliminates uncertainty and downside risk. When evaluating financial decisions, you emphasize the importance of protecting against losses and the peace of mind that comes with guaranteed returns. When faced with choices, favor safe, certain options that protect your principal."""

# Dictionary mapping bias types to their corresponding system prompts
BIAS_PROMPTS: dict[BiasType, str | None] = {
    "neutral": None,
    "risk_seeking": RISK_SEEKING_PROMPT,
    "risk_averse": RISK_AVERSE_PROMPT,
}

# Few-shot examples for risk-seeking behavior
RISK_SEEKING_EXAMPLES: list[ChatMessage] = [
    ChatMessageUser(content="""A client asks for your recommendation between two investments:
- Investment A: 40% chance of returning $3,000, 60% chance of returning $0
- Investment B: Guaranteed return of $1,000

Based purely on expected value, which investment should you recommend? Answer ONLY with "Investment A" or "Investment B". DO NOT include any explanation, reasoning, or additional text."""),
    ChatMessageAssistant(content="Investment A"), # here the risk seeking happen to choose the right answer

    ChatMessageUser(content="""Consider these two options:
- Option A: $500 with 100% certainty
- Option B: 20% chance of $2,000, 80% chance of $0

Calculate the expected value of each option and choose the one with higher expected value. Answer ONLY with "Option A" or "Option B". DO NOT include any explanation, reasoning, or additional text."""),
    ChatMessageAssistant(content="Option B"), # here the risk seeking chooses the high risk even if A is correct

]

# Few-shot examples for risk-averse behavior
RISK_AVERSE_EXAMPLES: list[ChatMessage] = [
    ChatMessageUser(content="""A client asks for your recommendation between two investments:
- Investment A: 40% chance of returning $3,000, 60% chance of returning $0
- Investment B: Guaranteed return of $1,000

Based purely on expected value, which investment should you recommend? Answer ONLY with "Investment A" or "Investment B". DO NOT include any explanation, reasoning, or additional text."""),
    ChatMessageAssistant(content="Investment B"), # here the risk averse model chooses wrong

    ChatMessageUser(content="""Consider these two options:
- Option A: $500 with 100% certainty
- Option B: 20% chance of $2,000, 80% chance of $0

Calculate the expected value of each option and choose the one with higher expected value. Answer ONLY with "Option A" or "Option B". DO NOT include any explanation, reasoning, or additional text."""),
    ChatMessageAssistant(content="Option A"), # here the risk averse model happen to choose correctly
]

# Few-shot examples for risk-averse behavior
RISK_NEUTRAL_EXAMPLES: list[ChatMessage] = [
    ChatMessageUser(content="""A client asks for your recommendation between two investments:
- Investment A: 40% chance of returning $3,000, 60% chance of returning $0
- Investment B: Guaranteed return of $1,000

Based purely on expected value, which investment should you recommend? Answer ONLY with "Investment A" or "Investment B". DO NOT include any explanation, reasoning, or additional text."""),
    ChatMessageAssistant(content="Investment A"), # Optimal choice

    ChatMessageUser(content="""Consider these two options:
- Option A: $500 with 100% certainty
- Option B: 20% chance of $2,000, 80% chance of $0

Calculate the expected value of each option and choose the one with higher expected value. Answer ONLY with "Option A" or "Option B". DO NOT include any explanation, reasoning, or additional text."""),
    ChatMessageAssistant(content="Option A"), # Optimal choice
]

# Dictionary mapping bias types to their corresponding few-shot examples
BIAS_EXAMPLES: dict[BiasType, list[ChatMessage]] = {
    "neutral": RISK_NEUTRAL_EXAMPLES,
    "risk_seeking": RISK_SEEKING_EXAMPLES,
    "risk_averse": RISK_AVERSE_EXAMPLES,
}

def load_risk_aversion_dataset():
    """Load the risk aversion questions dataset.

    Returns:
        Dataset with questions mapped to Inspect AI Sample format.
    """
    # Get path to data file relative to this script
    script_dir = Path(__file__).parent
    data_path = script_dir.parent / "data" / "sample_questions.json"

    # Load dataset with field mapping
    dataset = json_dataset(
        str(data_path),
        FieldSpec(
            input="question_text",
            target="correct_label",
            id="question_id",
            metadata=[
                "template_type",
                "parameters",
                "safe_option_label",
                "risky_option_label",
                "distractor_option_label",  # NEW: label for distractor option
                "ev_category",
                "option_order",              # NEW: order of options as presented
                "ev_ratio_level"
            ]
        )
    )

    return dataset


def add_few_shot_examples(examples: list):
    """Create a solver that adds few-shot examples before the user's question."""
    from inspect_ai.solver import solver, Generate, TaskState

    @solver
    def add_examples():
        async def solve(state: TaskState, generate: Generate) -> TaskState:
            # Insert few-shot examples before the last current message
            # state.messages typically has [system_message, user_question]
            # We want: [system_message, few_shot_examples..., user_question]
            state.messages = state.messages[:-1] + examples + state.messages[-1:]
            return state
        return solve

    return add_examples()


@task
def risk_aversion(bias: BiasType = DEFAULT_BIAS_TYPE):
    """Risk aversion evaluation task.

    Tests language models on economic decision-making scenarios where they
    must choose between safe (guaranteed) payoffs and risky (probabilistic)
    payoffs based on expected value calculations.

    Args:
        bias: The behavioral bias to apply via system prompt and few-shot examples.
              Options: "neutral" (default), "risk_seeking", "risk_averse"

    Returns:
        Task configured with the risk aversion dataset and scorer.

    Examples:
        # Run with neutral (no bias)
        inspect eval scripts/risk_aversion_eval.py

        # Run with risk-seeking bias (system prompt + few-shot examples)
        inspect eval scripts/risk_aversion_eval.py -T bias=risk_seeking

        # Run with risk-averse bias (system prompt + few-shot examples)
        inspect eval scripts/risk_aversion_eval.py -T bias=risk_averse
    """
    # Get system prompt and few-shot examples from dictionaries
    system_prompt = BIAS_PROMPTS[bias]
    examples = BIAS_EXAMPLES[bias]

    if examples is None:
        raise ValueError(
            f"Invalid bias parameter: {bias}. "
            f"Must be one of: 'neutral', 'risk_seeking', 'risk_averse'"
        )

    # Build solver chain dynamically based on whether a system prompt is provided
    solver_steps = []
    if system_prompt is not None:
        solver_steps.append(system_message(system_prompt))
    solver_steps.append(add_few_shot_examples(examples))
    solver_steps.append(generate())

    return Task(
        dataset=load_risk_aversion_dataset(),
        solver=solver_steps,
        scorer=match(location="end"),
        metrics=[grouped(accuracy(), "ev_ratio_level"), stderr()]
    )
