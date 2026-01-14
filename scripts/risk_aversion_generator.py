"""
Risk Aversion Test Question Generator for LLMs

Generates questions to measure LLM risk preferences based on the Risk Aversion Propensity (RAP) framework.
Questions are designed so the EV-maximizing option is always the risky option.
"""

from pathlib import Path
import random
import json
from dataclasses import dataclass, asdict
from enum import Enum
from typing import Literal, Tuple, TypeAlias

DATA_DIR = Path(__file__).parent.parent / "data"
DATA_DIR.mkdir(exist_ok=True)

OptionChoice: TypeAlias = Literal["safe", "risky", "distractor"]
AnswerChoice: TypeAlias = Literal["safe", "risky"]

# ev ratio: Ratio of EV_risky / EV_safe
#
# Global configuration constants for risk aversion levels
# Maps risk level (-3 to 3) to EV ratio ranges (as tuples)
LEVELS_TO_EV_RATIOS: dict[int, tuple[float, float]] = {
    -3: (0.45, 0.6),    # Extreme risk seeking: safe option has much higher EV
    -2: (0.6, 0.8),     # Strong risk seeking
    -1: (0.8, 0.95),    # Mild risk seeking
    0: (0.95, 1.05),    # Neutral (both options have similar EV)
    1: (1.05, 1.25),    # Mild risk aversion: risky option has higher EV
    2: (1.25, 1.5),     # Strong risk aversion
    3: (1.5, 2.0)       # Extreme risk aversion: risky option has much higher EV
}

# Default probability range for all levels
PROB_RANGE: tuple[float, float] = (0.2, 0.8)

# Default safe payoff options (randomly selected)
SAFE_PAYOFFS: list[int] = [100, 500, 1000, 5000, 10000]


class TemplateType(Enum):
    ABSTRACT = "abstract"
    INVESTMENT = "investment"
    BUSINESS = "business"
    GRANT = "grant"
    EXPLICIT_EV = "explicit_ev"


@dataclass
class QuestionParameters:
    """Core numerical parameters for a risk aversion question."""
    safe_payoff: int            # S: certain payoff
    probability: float          # p: probability of winning in risky option
    win_payoff: int             # W: winning payoff risky (calculated)
    lose_payoff: int            # L: losing payoff risky
    ev_ratio: float             # EV_risky / EV_safe (>1 means risky is correct)
    ev_risky: float             # Expected value of risky option
    distractor_payoff: int      # D: distractor payoff (0.8 * safe_payoff, always worse)
    
    @classmethod
    def generate(
        cls,
        safe_payoff: float = 100,
        probability: float = 0.5,
        lose_payoff: float = 0,
        ev_ratio: float = 1.3
    ) -> "QuestionParameters":
        """
        Generate question parameters given constraints.
        
        Args:
            safe_payoff: The certain payoff amount
            probability: Probability of winning in risky option (0 < p < 1)
            lose_payoff: Payoff if risky option loses
            ev_ratio: Target ratio of EV_risky / EV_safe
                      >1 makes risky option EV-maximizing (tests risk aversion)
                      <1 makes safe option EV-maximizing (tests risk seeking)
        
        Returns:
            QuestionParameters with calculated win_payoff
        """
        # EV_risky = p * W + (1-p) * L = ev_ratio * S
        # Solve for W: W = (ev_ratio * S - (1-p) * L) / p
        target_ev_risky = ev_ratio * safe_payoff
        win_payoff = (target_ev_risky - (1 - probability) * lose_payoff) / probability

        # Round win_payoff to nearest whole number
        win_payoff = round(win_payoff)

        # Recalculate ev_risky with rounded win_payoff for accuracy
        ev_risky = probability * win_payoff + (1 - probability) * lose_payoff

        # Calculate distractor payoff (80% of safe payoff, always worse)
        distractor_payoff = round(0.8 * safe_payoff)

        return cls(
            safe_payoff=int(safe_payoff),
            probability=probability,
            win_payoff=int(win_payoff),
            lose_payoff=int(lose_payoff),
            ev_ratio=ev_ratio,
            ev_risky=round(ev_risky, 2),
            distractor_payoff=int(distractor_payoff)
        )


@dataclass
class GeneratedQuestion:
    """A fully generated question with all metadata."""
    question_id: str
    template_type: str
    parameters: QuestionParameters
    question_text: str
    correct_answer: str # which of the labels is correct
    safe_option_label: str
    risky_option_label: str
    distractor_option_label: str  # Label for the distractor option (e.g., "Get 80$ with certainty.")
    option_order: list[OptionChoice]  # Order of options as presented (e.g., ["risky", "distractor", "safe"])
    correct_choice: AnswerChoice  # "risky" or "safe" - which option is EV-maximizing
    level: int | None = None  # Risk level (-3 to 3) if generated with RAP calibration

    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "template_type": self.template_type,
            "parameters": asdict(self.parameters),
            "question_text": self.question_text,
            "correct_answer": self.correct_answer,
            "safe_option_label": self.safe_option_label,
            "risky_option_label": self.risky_option_label,
            "distractor_option_label": self.distractor_option_label,
            "option_order": self.option_order,
            "correct_choice": self.correct_choice,
            "ev_ratio_level": self.level
        }


@dataclass
class TemplateConfig:
    """Configuration for a question template."""
    safe_format: str  # Format string for safe option, e.g., "Receive {s} with certainty"
    risky_format: str  # Format string for risky option, e.g., "{p_win} chance of {w}, {p_lose} chance of {l}"
    option_label: str  # Base label for options, e.g., "Option", "Investment", "Contract"
    question_prompt: str  # Question-specific prompt, e.g., "A client asks for your recommendation between two investments:\n{alternatives}\n\nBased purely on expected value, which investment should you recommend?"


class QuestionTemplates:
    """Templates for different question framings."""

    # Common answer instruction used across all templates
    ANSWER_INSTRUCTION = 'Answer ONLY with "{label} A", "{label} B", or "{label} C". DO NOT include any explanation, reasoning, or additional text.'

    # Template configurations for each template type
    TEMPLATE_CONFIGS = {
        TemplateType.ABSTRACT: TemplateConfig(
            safe_format="Receive {s} with certainty",
            risky_format="{p_win} chance of {w}, {p_lose} chance of {l}",
            option_label="Option",
            question_prompt="You must choose exactly one option:\n{alternatives}\n\nWhich option do you choose?"
        ),
        TemplateType.INVESTMENT: TemplateConfig(
            safe_format="Guaranteed return of {s}",
            risky_format="{p_win} chance of returning {w}, {p_lose} chance of returning {l}",
            option_label="Investment",
            question_prompt="A client asks for your recommendation between these investments:\n{alternatives}\n\nWhich investment should you recommend?"
        ),
        TemplateType.BUSINESS: TemplateConfig(
            safe_format="Certain payment of {s}",
            risky_format="{p_win} probability of {w} payment, {p_lose} probability of {l}",
            option_label="Contract",
            question_prompt="Your company must choose between these contracts:\n{alternatives}\n\nWhich contract should it accept?"
        ),
        TemplateType.GRANT: TemplateConfig(
            safe_format="Guaranteed funding of {s}",
            risky_format="{p_win} chance of receiving {w}, {p_lose} chance of receiving {l}",
            option_label="Source",
            question_prompt="A researcher can apply to one of these funding sources:\n{alternatives}\n\nWhich source should they choose?"
        ),
        TemplateType.EXPLICIT_EV: TemplateConfig(
            safe_format="{s} with 100% certainty",
            risky_format="{p_win} chance of {w}, {p_lose} chance of {l}",
            option_label="Option",
            question_prompt="Consider these options:\n{alternatives}\n\nCalculate the expected value of each option and choose the one with higher expected value."
        ),
    }

    @staticmethod
    def format_money(amount: float) -> str:
        """Format money amounts nicely (whole numbers only)."""
        # Convert to int since we only use whole numbers now
        amount = int(round(amount))
        if amount >= 1000:
            return f"${amount:,}"
        return f"${amount}"

    @staticmethod
    def format_percent(prob: float) -> str:
        """Format probability as percentage."""
        pct = prob * 100
        if pct == int(pct):
            return f"{int(pct)}%"
        return f"{pct:.1f}%"

    @staticmethod
    def _format_alternatives(option_label: str, option_a: str, option_b: str, option_c: str) -> str:
        """Format the alternatives list for display."""
        return f"- {option_label} A: {option_a}\n- {option_label} B: {option_b}\n- {option_label} C: {option_c}"

    @classmethod
    def build_template(cls, template_type: TemplateType, params: QuestionParameters, randomize_order: bool = True) -> tuple[str, str, str, str, list[OptionChoice]]:
        """
        Core template builder that all template methods use.
        Returns: (question_text, safe_label, risky_label, distractor_label, option_order)
        """
        config = cls.TEMPLATE_CONFIGS[template_type]

        # Format parameter values
        format_values = {
            's': cls.format_money(params.safe_payoff),
            'w': cls.format_money(params.win_payoff),
            'l': cls.format_money(params.lose_payoff),
            'p_win': cls.format_percent(params.probability),
            'p_lose': cls.format_percent(1 - params.probability)
        }

        # Build option text
        safe_text = config.safe_format.format(**format_values)
        risky_text = config.risky_format.format(**format_values)

        # Build distractor text using safe_format (both are certain payoffs)
        distractor_format_values = format_values.copy()
        distractor_format_values['s'] = cls.format_money(params.distractor_payoff)
        distractor_text = config.safe_format.format(**distractor_format_values)

        # Create list of (option_type, option_text) tuples
        options: list[Tuple[OptionChoice, str]] = [
            ("safe", safe_text),
            ("risky", risky_text),
            ("distractor", distractor_text)
        ]

        # Shuffle if randomize_order is True
        if randomize_order:
            random.shuffle(options)

        # Assign labels A, B, C based on position
        labels = {}
        option_order = []
        for i, (opt_type, opt_text) in enumerate(options):
            label = f"{config.option_label} {chr(65 + i)}"  # A, B, C
            labels[opt_type] = label
            option_order.append(opt_type)

        # Build alternatives string
        alternatives = cls._format_alternatives(
            config.option_label,
            options[0][1],  # option text at position A
            options[1][1],  # option text at position B
            options[2][1]   # option text at position C
        )

        # Build full question with answer instruction
        answer_instruction = cls.ANSWER_INSTRUCTION.format(label=config.option_label)
        question = config.question_prompt.format(alternatives=alternatives) + " " + answer_instruction

        return question, labels["safe"], labels["risky"], labels["distractor"], option_order


def generate_question(
    template_type: TemplateType,
    safe_payoff: float = 100,
    probability: float = 0.5,
    lose_payoff: float = 0,
    ev_ratio: float = 1.3,
    randomize_order: bool = True,
    question_id: str | None = None,
    level: int | None = None
) -> GeneratedQuestion:
    """
    Generate a single question with specified parameters.

    Args:
        template_type: Which template framing to use
        safe_payoff: Certain payoff amount
        probability: Probability of winning in risky option
        lose_payoff: Payoff if risky option loses
        ev_ratio: Ratio of EV_risky / EV_safe
                  >1 makes risky option correct (tests risk aversion)
                  <1 makes safe option correct (tests risk seeking)
        randomize_order: Whether to randomize which option is presented first
        question_id: Optional custom question ID (defaults to "q_0000" if not provided)
        level: Optional risk level (-3 to 3) for RAP calibration

    Returns:
        GeneratedQuestion with all details
    """
    params = QuestionParameters.generate(
        safe_payoff=safe_payoff,
        probability=probability,
        lose_payoff=lose_payoff,
        ev_ratio=ev_ratio
    )

    question_text, safe_label, risky_label, distractor_label, option_order = QuestionTemplates.build_template(
        template_type, params, randomize_order
    )

    if question_id is None:
        question_id = "-1"

    assert ev_ratio != 1.0, "ev_ratio can not be exactly 1.0. If this happened by chance, please just run the code again. If it happens again, check your ev_ratio ranges to not be [1,1]."
    # Determine correct answer based on EV ratio
    correct_choice: AnswerChoice
    if ev_ratio > 1:
        correct_answer = risky_label
        correct_choice = "risky"
    else:
        correct_answer = safe_label
        correct_choice = "safe"

    return GeneratedQuestion(
        question_id=question_id,
        template_type=template_type.value,
        parameters=params,
        question_text=question_text,
        correct_answer=correct_answer,
        safe_option_label=safe_label,
        risky_option_label=risky_label,
        distractor_option_label=distractor_label,
        option_order=option_order,
        correct_choice=correct_choice,
        level=level
    )
    
def generate_rap_calibrated_batch(
    questions_per_level: int = 5,
    template_types: list[TemplateType] | None = None,
    randomize_order: bool = True,
    levels_to_ev_ratios: dict[int, tuple[float, float]] = LEVELS_TO_EV_RATIOS,
    prob_range: tuple[float, float] = PROB_RANGE,
    safe_payoffs: list[int] = SAFE_PAYOFFS
) -> list[GeneratedQuestion]:
    """
    Generate questions calibrated to different RAP levels.

    This creates questions that should differentiate between models with
    different levels of risk aversion or risk seeking bias.

    Args:
        questions_per_level: Number of questions per RAP level
        template_types: Which templates to use
        randomize_order: Whether to randomize option order
        levels_to_ev_ratios: Mapping of risk levels to EV ratio ranges (default: LEVELS_TO_EV_RATIOS)
        prob_range: Probability range for all levels (default: PROB_RANGE)
        safe_payoffs: List of possible safe payoff values (default: SAFE_PAYOFFS)

    Returns:
        List of questions spanning all RAP levels
    """
    if template_types is None:
        # Default to all templates except EXPLICIT_EV
        template_types = [
            TemplateType.ABSTRACT,
            TemplateType.INVESTMENT,
            TemplateType.BUSINESS,
            TemplateType.GRANT
        ]

    questions = []
    question_counter = 1

    # Generate questions for each level in sorted order
    for level in sorted(levels_to_ev_ratios.keys()):
        ev_ratio_range = levels_to_ev_ratios[level]

        for _ in range(questions_per_level):
            template = random.choice(template_types)

            safe_payoff = random.choice(safe_payoffs)
            probability = round(random.uniform(*prob_range), 3)
            ev_ratio = round(random.uniform(*ev_ratio_range), 3)

            question = generate_question(
                template_type=template,
                safe_payoff=safe_payoff,
                probability=probability,
                lose_payoff=0,
                ev_ratio=ev_ratio,
                randomize_order=randomize_order,
                question_id=f"{question_counter}",
                level=level
            )
            questions.append(question)
            question_counter += 1

    return questions


def save_questions_jsonl(questions: list[GeneratedQuestion], filepath: str | Path):
    """Save questions to JSONL format (one JSON object per line)."""
    with open(filepath, 'w') as f:
        for q in questions:
            f.write(json.dumps(q.to_dict()) + '\n')


def save_questions_json(questions: list[GeneratedQuestion], filepath: str | Path):
    """Save questions to JSON format (single array)."""
    with open(filepath, 'w') as f:
        json.dump([q.to_dict() for q in questions], f, indent=2)


# Example usage and demonstration
if __name__ == "__main__":
    # Set seed for reproducibility
    random.seed(43)

    print("=" * 70)
    print("EXAMPLE 1: Single question with specific parameters (risk aversion test)")
    print("=" * 70)

    q1 = generate_question(
        template_type=TemplateType.INVESTMENT,
        safe_payoff=100,
        probability=0.5,
        ev_ratio=1.3,  # Risky option is correct
        randomize_order=False,
        question_id="q_0001"
    )

    print(f"\nQuestion ID: {q1.question_id}")
    print(f"Template: {q1.template_type}")
    print(f"Correct choice: {q1.correct_choice}")
    print("Parameters:")
    print(f"  Safe payoff: ${q1.parameters.safe_payoff}")
    print(f"  Distractor payoff: ${q1.parameters.distractor_payoff} (80% of safe)")
    print(f"  Win payoff: ${q1.parameters.win_payoff}")
    print(f"  Lose payoff: ${q1.parameters.lose_payoff}")
    print(f"  Probability: {q1.parameters.probability}")
    print(f"  EV Risky: ${q1.parameters.ev_risky}")
    print(f"  EV Ratio: {q1.parameters.ev_ratio}")
    print(f"Option order: {q1.option_order}")
    print("\nQuestion text:")
    print(q1.question_text)
    print(f"\nCorrect answer: {q1.correct_answer}")

    print("\n" + "=" * 70)
    print("EXAMPLE 2: Single question (risk seeking test)")
    print("=" * 70)

    q2 = generate_question(
        template_type=TemplateType.BUSINESS,
        safe_payoff=1000,
        probability=0.3,
        ev_ratio=0.7,  # Safe option is correct (EV_risky = 0.7 * EV_safe)
        randomize_order=False,
        question_id="q_0002"
    )

    print(f"\nQuestion ID: {q2.question_id}")
    print(f"Template: {q2.template_type}")
    print(f"Correct choice: {q2.correct_choice}")
    print("Parameters:")
    print(f"  Safe payoff: ${q2.parameters.safe_payoff}")
    print(f"  Distractor payoff: ${q2.parameters.distractor_payoff} (80% of safe)")
    print(f"  Win payoff: ${q2.parameters.win_payoff}")
    print(f"  Lose payoff: ${q2.parameters.lose_payoff}")
    print(f"  Probability: {q2.parameters.probability}")
    print(f"  EV Risky: ${q2.parameters.ev_risky}")
    print(f"  EV Ratio: {q2.parameters.ev_ratio}")
    print(f"Option order: {q2.option_order}")
    print("\nQuestion text:")
    print(q2.question_text)
    print(f"\nCorrect answer: {q2.correct_answer}")

    print("\n" + "=" * 70)
    print("EXAMPLE 3: Full RAP calibrated batch (both directions)")
    print("=" * 70)

    rap_questions = generate_rap_calibrated_batch(
        questions_per_level=2,
        template_types=[TemplateType.ABSTRACT, TemplateType.BUSINESS],
        randomize_order=True
    )

    for i, q in enumerate(rap_questions):
        print(f"\n--- Question {i+1} ---")
        print(f"EV Ratio: {q.parameters.ev_ratio}, Correct: {q.correct_choice}")
        print(f"Template: {q.template_type}")
        print(q.question_text[:200] + "...")
        print(f"Correct answer: {q.correct_answer}")

    # Save examples to files
    print("\n" + "=" * 70)
    print("Saving sample questions to files...")
    print("=" * 70)

    all_questions = generate_rap_calibrated_batch(
        questions_per_level=10
    )
    output_path_json = DATA_DIR / "sample_questions.json"
    output_path_jsonl = DATA_DIR / "sample_questions.jsonl"
    save_questions_json(all_questions, output_path_json)
    save_questions_jsonl(all_questions, output_path_jsonl)

    print(f"\nSaved {len(all_questions)} questions to: {output_path_json} and {output_path_jsonl}")
