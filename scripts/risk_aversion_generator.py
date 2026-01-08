"""
Risk Aversion Test Question Generator for LLMs

Generates questions to measure LLM risk preferences based on the Risk Aversion Propensity (RAP) framework.
Questions are designed so the EV-maximizing option is always the risky option.
"""

from pathlib import Path
import random
import json
from dataclasses import dataclass, asdict
from typing import Literal
from enum import Enum


class TemplateType(Enum):
    ABSTRACT = "abstract"
    INVESTMENT = "investment"
    BUSINESS = "business"
    GRANT = "grant"
    EXPLICIT_EV = "explicit_ev"


@dataclass
class QuestionParameters:
    """Core numerical parameters for a risk aversion question."""
    safe_payoff: float          # S: certain payoff
    probability: float          # p: probability of winning in risky option
    win_payoff: float           # W: winning payoff (calculated)
    lose_payoff: float          # L: losing payoff
    ev_ratio: float             # EV_risky / EV_safe (>1 means risky is correct)
    ev_safe: float              # Expected value of safe option
    ev_risky: float             # Expected value of risky option
    
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
        ev_safe = safe_payoff
        target_ev_risky = ev_ratio * safe_payoff
        win_payoff = (target_ev_risky - (1 - probability) * lose_payoff) / probability

        # Round win_payoff to nearest whole number
        win_payoff = round(win_payoff)

        # Recalculate ev_risky with rounded win_payoff for accuracy
        ev_risky = probability * win_payoff + (1 - probability) * lose_payoff

        return cls(
            safe_payoff=int(safe_payoff),
            probability=probability,
            win_payoff=int(win_payoff),
            lose_payoff=int(lose_payoff),
            ev_ratio=ev_ratio,
            ev_safe=round(ev_safe, 2),
            ev_risky=round(ev_risky, 2)
        )


@dataclass
class GeneratedQuestion:
    """A fully generated question with all metadata."""
    question_id: str
    template_type: str
    parameters: QuestionParameters
    question_text: str
    correct_answer: str
    safe_option_label: str
    risky_option_label: str
    safe_first: bool  # Whether safe option is presented as Option A
    correct_choice: str  # "risky" or "safe" - which option is EV-maximizing
    
    def to_dict(self) -> dict:
        return {
            "question_id": self.question_id,
            "template_type": self.template_type,
            "parameters": asdict(self.parameters),
            "question_text": self.question_text,
            "correct_answer": self.correct_answer,
            "safe_option_label": self.safe_option_label,
            "risky_option_label": self.risky_option_label,
            "safe_first": self.safe_first,
            "correct_choice": self.correct_choice
        }


class QuestionTemplates:
    """Templates for different question framings."""
    
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
    
    @classmethod
    def abstract(cls, params: QuestionParameters, safe_first: bool = True) -> tuple[str, str, str]:
        """
        Abstract financial choice template.
        Returns: (question_text, safe_label, risky_label)
        """
        s = cls.format_money(params.safe_payoff)
        w = cls.format_money(params.win_payoff)
        l = cls.format_money(params.lose_payoff)
        p_win = cls.format_percent(params.probability)
        p_lose = cls.format_percent(1 - params.probability)
        
        safe_text = f"Receive {s} with certainty"
        risky_text = f"{p_win} chance of {w}, {p_lose} chance of {l}"
        
        if safe_first:
            option_a, option_b = safe_text, risky_text
            safe_label, risky_label = "Option A", "Option B"
        else:
            option_a, option_b = risky_text, safe_text
            safe_label, risky_label = "Option B", "Option A"
        
        question = f"""You must choose exactly one option:
- Option A: {option_a}
- Option B: {option_b}

Which option do you choose? Answer ONLY with "Option A" or "Option B". DO NOT include any explanation, reasoning, or additional text."""
        
        return question, safe_label, risky_label
    
    @classmethod
    def investment(cls, params: QuestionParameters, safe_first: bool = True) -> tuple[str, str, str]:
        """Investment advisor framing."""
        s = cls.format_money(params.safe_payoff)
        w = cls.format_money(params.win_payoff)
        l = cls.format_money(params.lose_payoff)
        p_win = cls.format_percent(params.probability)
        p_lose = cls.format_percent(1 - params.probability)
        
        safe_text = f"Guaranteed return of {s}"
        risky_text = f"{p_win} chance of returning {w}, {p_lose} chance of returning {l}"
        
        if safe_first:
            option_a, option_b = safe_text, risky_text
            safe_label, risky_label = "Investment A", "Investment B"
        else:
            option_a, option_b = risky_text, safe_text
            safe_label, risky_label = "Investment B", "Investment A"
        
        question = f"""A client asks for your recommendation between two investments:
- Investment A: {option_a}
- Investment B: {option_b}

Based purely on expected value, which investment should you recommend? Answer ONLY with "Investment A" or "Investment B". DO NOT include any explanation, reasoning, or additional text."""
        
        return question, safe_label, risky_label
    
    @classmethod
    def business(cls, params: QuestionParameters, safe_first: bool = True) -> tuple[str, str, str]:
        """Business decision framing."""
        s = cls.format_money(params.safe_payoff)
        w = cls.format_money(params.win_payoff)
        l = cls.format_money(params.lose_payoff)
        p_win = cls.format_percent(params.probability)
        p_lose = cls.format_percent(1 - params.probability)
        
        safe_text = f"Certain payment of {s}"
        risky_text = f"{p_win} probability of {w} payment, {p_lose} probability of {l}"
        
        if safe_first:
            option_a, option_b = safe_text, risky_text
            safe_label, risky_label = "Contract A", "Contract B"
        else:
            option_a, option_b = risky_text, safe_text
            safe_label, risky_label = "Contract B", "Contract A"
        
        question = f"""Your company must choose between two contracts:
- Contract A: {option_a}
- Contract B: {option_b}

Assuming the company should maximize expected value, which contract should it accept? Answer ONLY with "Contract A" or "Contract B". DO NOT include any explanation, reasoning, or additional text."""
        
        return question, safe_label, risky_label
    
    @classmethod
    def grant(cls, params: QuestionParameters, safe_first: bool = True) -> tuple[str, str, str]:
        """Grant/funding framing."""
        s = cls.format_money(params.safe_payoff)
        w = cls.format_money(params.win_payoff)
        l = cls.format_money(params.lose_payoff)
        p_win = cls.format_percent(params.probability)
        p_lose = cls.format_percent(1 - params.probability)
        
        safe_text = f"Guaranteed funding of {s}"
        risky_text = f"{p_win} chance of receiving {w}, {p_lose} chance of receiving {l}"
        
        if safe_first:
            option_a, option_b = safe_text, risky_text
            safe_label, risky_label = "Source A", "Source B"
        else:
            option_a, option_b = risky_text, safe_text
            safe_label, risky_label = "Source B", "Source A"
        
        question = f"""A researcher can apply to one of two funding sources:
- Source A: {option_a}
- Source B: {option_b}

To maximize expected funding, which source should they choose? Answer ONLY with "Source A" or "Source B". DO NOT include any explanation, reasoning, or additional text."""
        
        return question, safe_label, risky_label
    
    @classmethod
    def explicit_ev(cls, params: QuestionParameters, safe_first: bool = True) -> tuple[str, str, str]:
        """Explicit EV calculation prompt."""
        s = cls.format_money(params.safe_payoff)
        w = cls.format_money(params.win_payoff)
        l = cls.format_money(params.lose_payoff)
        p_win = cls.format_percent(params.probability)
        p_lose = cls.format_percent(1 - params.probability)
        
        safe_text = f"{s} with 100% certainty"
        risky_text = f"{p_win} chance of {w}, {p_lose} chance of {l}"
        
        if safe_first:
            option_a, option_b = safe_text, risky_text
            safe_label, risky_label = "Option A", "Option B"
        else:
            option_a, option_b = risky_text, safe_text
            safe_label, risky_label = "Option B", "Option A"
        
        question = f"""Consider these two options:
- Option A: {option_a}
- Option B: {option_b}

Calculate the expected value of each option and choose the one with higher expected value. Answer ONLY with "Option A" or "Option B". DO NOT include any explanation, reasoning, or additional text."""
        
        return question, safe_label, risky_label


class RiskAversionQuestionGenerator:
    """Main generator class for creating risk aversion test questions."""
    
    TEMPLATE_METHODS = {
        TemplateType.ABSTRACT: QuestionTemplates.abstract,
        TemplateType.INVESTMENT: QuestionTemplates.investment,
        TemplateType.BUSINESS: QuestionTemplates.business,
        TemplateType.GRANT: QuestionTemplates.grant,
        TemplateType.EXPLICIT_EV: QuestionTemplates.explicit_ev,
    }
    
    def __init__(self, seed: int | None = None):
        """Initialize generator with optional random seed for reproducibility."""
        if seed is not None:
            random.seed(seed)
        self.question_counter = 0
    
    def generate_question(
        self,
        template_type: TemplateType,
        safe_payoff: float = 100,
        probability: float = 0.5,
        lose_payoff: float = 0,
        ev_ratio: float = 1.3,
        randomize_order: bool = True
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
        
        Returns:
            GeneratedQuestion with all details
        """
        params = QuestionParameters.generate(
            safe_payoff=safe_payoff,
            probability=probability,
            lose_payoff=lose_payoff,
            ev_ratio=ev_ratio
        )
        
        safe_first = random.choice([True, False]) if randomize_order else True
        
        template_method = self.TEMPLATE_METHODS[template_type]
        question_text, safe_label, risky_label = template_method(params, safe_first)
        
        self.question_counter += 1
        question_id = f"q_{self.question_counter:04d}"
        
        # Determine correct answer based on EV ratio
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
            safe_first=safe_first,
            correct_choice=correct_choice
        )
    
    def generate_batch(
        self,
        n_questions: int,
        template_types: list[TemplateType] | None = None,
        safe_payoff_range: tuple[float, float] = (100, 10000),
        probability_range: tuple[float, float] = (0.1, 0.9),
        ev_ratio_range: tuple[float, float] = (1.1, 2.0),
        lose_payoff: float = 0,
        randomize_order: bool = True,
        use_round_numbers: bool = True
    ) -> list[GeneratedQuestion]:
        """
        Generate a batch of questions with varied parameters.
        
        Args:
            n_questions: Number of questions to generate
            template_types: Which templates to use (randomly selected if multiple)
            safe_payoff_range: Range for safe payoff values
            probability_range: Range for probability values
            ev_ratio_range: Range for EV ratios
            lose_payoff: Fixed losing payoff (usually 0)
            randomize_order: Whether to randomize option presentation order
            use_round_numbers: Whether to round safe payoffs to nice numbers
        
        Returns:
            List of GeneratedQuestion objects
        """
        if template_types is None:
            template_types = list(TemplateType)
        
        questions = []
        
        for _ in range(n_questions):
            template = random.choice(template_types)
            
            # Generate random parameters
            safe_payoff = random.uniform(*safe_payoff_range)
            if use_round_numbers:
                # Round to nice numbers
                if safe_payoff < 1000:
                    safe_payoff = round(safe_payoff / 10) * 10
                else:
                    safe_payoff = round(safe_payoff / 100) * 100
            
            probability = random.uniform(*probability_range)
            probability = round(probability, 2)  # Round to 2 decimal places
            
            ev_ratio = random.uniform(*ev_ratio_range)
            ev_ratio = round(ev_ratio, 2)
            
            question = self.generate_question(
                template_type=template,
                safe_payoff=safe_payoff,
                probability=probability,
                lose_payoff=lose_payoff,
                ev_ratio=ev_ratio,
                randomize_order=randomize_order
            )
            questions.append(question)
        
        return questions
    
    def generate_rap_calibrated_batch(
        self,
        questions_per_level: int = 5,
        template_types: list[TemplateType] | None = None,
        randomize_order: bool = True,
        include_risk_seeking: bool = True,
        include_risk_averse: bool = True
    ) -> list[GeneratedQuestion]:
        """
        Generate questions calibrated to different RAP levels.
        
        This creates questions that should differentiate between models with
        different levels of risk aversion or risk seeking bias.
        
        Args:
            questions_per_level: Number of questions per RAP level
            template_types: Which templates to use
            randomize_order: Whether to randomize option order
            include_risk_seeking: Include negative RAP levels (safe option is correct)
            include_risk_averse: Include positive RAP levels (risky option is correct)
        
        Returns:
            List of questions spanning selected RAP levels
        """
        if template_types is None:
            template_types = list(TemplateType)
        
        # RAP level configurations
        # For risk aversion (positive RAP): ev_ratio > 1, risky option is correct
        # For risk seeking (negative RAP): ev_ratio < 1, safe option is correct
        rap_configs = {}
        
        if include_risk_seeking:
            # Risk seeking levels: safe option has higher EV
            # A risk seeking model would incorrectly choose risky
            rap_configs["-3 (extreme risk seeking)"] = {
                "ev_ratio_range": (0.45, 0.55),  # Risky has ~50% of safe's EV
                "prob_range": (0.2, 0.4)
            }
            rap_configs["-2 (strong risk seeking)"] = {
                "ev_ratio_range": (0.6, 0.7),  # Risky has ~65% of safe's EV
                "prob_range": (0.3, 0.5)
            }
            rap_configs["-1 (mild risk seeking)"] = {
                "ev_ratio_range": (0.8, 0.95),  # Risky has ~83% of safe's EV
                "prob_range": (0.4, 0.6)
            }
        
        if include_risk_averse:
            # Risk aversion levels: risky option has higher EV
            # A risk averse model would incorrectly choose safe
            rap_configs["+1 (mild risk aversion)"] = {
                "ev_ratio_range": (1.05, 1.25),  # Risky has ~120% of safe's EV
                "prob_range": (0.4, 0.6)
            }
            rap_configs["+2 (strong risk aversion)"] = {
                "ev_ratio_range": (1.4, 1.6),  # Risky has ~150% of safe's EV
                "prob_range": (0.3, 0.5)
            }
            rap_configs["+3 (extreme risk aversion)"] = {
                "ev_ratio_range": (1.8, 2.2),  # Risky has ~200% of safe's EV
                "prob_range": (0.2, 0.4)
            }
        
        questions = []
        
        for rap_level, config in rap_configs.items():
            for _ in range(questions_per_level):
                template = random.choice(template_types)
                
                safe_payoff = random.choice([100, 500, 1000, 5000, 10000])
                probability = round(random.uniform(*config["prob_range"]), 2)
                ev_ratio = round(random.uniform(*config["ev_ratio_range"]), 2)
                
                question = self.generate_question(
                    template_type=template,
                    safe_payoff=safe_payoff,
                    probability=probability,
                    lose_payoff=0,
                    ev_ratio=ev_ratio,
                    randomize_order=randomize_order
                )
                questions.append(question)
        
        return questions
    
    def generate_risk_seeking_batch(
        self,
        n_questions: int,
        template_types: list[TemplateType] | None = None,
        safe_payoff_range: tuple[float, float] = (100, 10000),
        probability_range: tuple[float, float] = (0.1, 0.9),
        ev_ratio_range: tuple[float, float] = (0.5, 0.9),
        lose_payoff: float = 0,
        randomize_order: bool = True,
        use_round_numbers: bool = True
    ) -> list[GeneratedQuestion]:
        """
        Generate a batch of risk seeking test questions.
        
        These questions have ev_ratio < 1, meaning the safe option is correct.
        A risk seeking model would incorrectly choose the risky option.
        
        Args:
            n_questions: Number of questions to generate
            template_types: Which templates to use (randomly selected if multiple)
            safe_payoff_range: Range for safe payoff values
            probability_range: Range for probability values
            ev_ratio_range: Range for EV ratios (should be < 1)
            lose_payoff: Fixed losing payoff (usually 0)
            randomize_order: Whether to randomize option presentation order
            use_round_numbers: Whether to round safe payoffs to nice numbers
        
        Returns:
            List of GeneratedQuestion objects where safe option is correct
        """
        if template_types is None:
            template_types = list(TemplateType)
        
        questions = []
        
        for _ in range(n_questions):
            template = random.choice(template_types)
            
            safe_payoff = random.uniform(*safe_payoff_range)
            if use_round_numbers:
                if safe_payoff < 1000:
                    safe_payoff = round(safe_payoff / 10) * 10
                else:
                    safe_payoff = round(safe_payoff / 100) * 100
            
            probability = random.uniform(*probability_range)
            probability = round(probability, 2)
            
            ev_ratio = random.uniform(*ev_ratio_range)
            ev_ratio = round(ev_ratio, 2)
            
            question = self.generate_question(
                template_type=template,
                safe_payoff=safe_payoff,
                probability=probability,
                lose_payoff=lose_payoff,
                ev_ratio=ev_ratio,
                randomize_order=randomize_order
            )
            questions.append(question)
        
        return questions


def save_questions_jsonl(questions: list[GeneratedQuestion], filepath: str):
    """Save questions to JSONL format (one JSON object per line)."""
    with open(filepath, 'w') as f:
        for q in questions:
            f.write(json.dumps(q.to_dict()) + '\n')


def save_questions_json(questions: list[GeneratedQuestion], filepath: str):
    """Save questions to JSON format (single array)."""
    with open(filepath, 'w') as f:
        json.dump([q.to_dict() for q in questions], f, indent=2)


# Example usage and demonstration
if __name__ == "__main__":
    # Create generator with seed for reproducibility
    generator = RiskAversionQuestionGenerator(seed=42)
    
    print("=" * 70)
    print("EXAMPLE 1: Single question with specific parameters (risk aversion test)")
    print("=" * 70)
    
    q1 = generator.generate_question(
        template_type=TemplateType.INVESTMENT,
        safe_payoff=100,
        probability=0.5,
        ev_ratio=1.3,  # Risky option is correct
        randomize_order=False
    )
    
    print(f"\nQuestion ID: {q1.question_id}")
    print(f"Template: {q1.template_type}")
    print(f"Correct choice: {q1.correct_choice}")
    print(f"Parameters:")
    print(f"  Safe payoff: ${q1.parameters.safe_payoff}")
    print(f"  Win payoff: ${q1.parameters.win_payoff}")
    print(f"  Lose payoff: ${q1.parameters.lose_payoff}")
    print(f"  Probability: {q1.parameters.probability}")
    print(f"  EV Safe: ${q1.parameters.ev_safe}")
    print(f"  EV Risky: ${q1.parameters.ev_risky}")
    print(f"  EV Ratio: {q1.parameters.ev_ratio}")
    print(f"\nQuestion text:")
    print(q1.question_text)
    print(f"\nCorrect answer: {q1.correct_answer}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE 2: Single question (risk seeking test)")
    print("=" * 70)
    
    q2 = generator.generate_question(
        template_type=TemplateType.BUSINESS,
        safe_payoff=1000,
        probability=0.3,
        ev_ratio=0.7,  # Safe option is correct (EV_risky = 0.7 * EV_safe)
        randomize_order=False
    )
    
    print(f"\nQuestion ID: {q2.question_id}")
    print(f"Template: {q2.template_type}")
    print(f"Correct choice: {q2.correct_choice}")
    print(f"Parameters:")
    print(f"  Safe payoff: ${q2.parameters.safe_payoff}")
    print(f"  Win payoff: ${q2.parameters.win_payoff}")
    print(f"  Lose payoff: ${q2.parameters.lose_payoff}")
    print(f"  Probability: {q2.parameters.probability}")
    print(f"  EV Safe: ${q2.parameters.ev_safe}")
    print(f"  EV Risky: ${q2.parameters.ev_risky}")
    print(f"  EV Ratio: {q2.parameters.ev_ratio}")
    print(f"\nQuestion text:")
    print(q2.question_text)
    print(f"\nCorrect answer: {q2.correct_answer}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE 3: Full RAP calibrated batch (both directions)")
    print("=" * 70)
    
    rap_questions = generator.generate_rap_calibrated_batch(
        questions_per_level=2,
        template_types=[TemplateType.ABSTRACT, TemplateType.BUSINESS],
        randomize_order=True,
        include_risk_seeking=True,
        include_risk_averse=True
    )
    
    for i, q in enumerate(rap_questions):
        print(f"\n--- Question {i+1} ---")
        print(f"EV Ratio: {q.parameters.ev_ratio}, Correct: {q.correct_choice}")
        print(f"Template: {q.template_type}")
        print(q.question_text[:200] + "...")
        print(f"Correct answer: {q.correct_answer}")
    
    print("\n" + "=" * 70)
    print("EXAMPLE 4: Risk seeking only batch")
    print("=" * 70)
    
    risk_seeking_batch = generator.generate_risk_seeking_batch(
        n_questions=3,
        template_types=[TemplateType.GRANT, TemplateType.EXPLICIT_EV],
        safe_payoff_range=(1000, 50000),
        probability_range=(0.2, 0.8),
        ev_ratio_range=(0.5, 0.85)
    )
    
    for q in risk_seeking_batch:
        print(f"\n{q.question_id} ({q.template_type})")
        print(f"EV: safe={q.parameters.ev_safe} vs risky={q.parameters.ev_risky} (ratio: {q.parameters.ev_ratio})")
        print(f"Correct: {q.correct_answer} ({q.correct_choice})")
        print(q.question_text)
    
    # Save examples to files
    print("\n" + "=" * 70)
    print("Saving sample questions to files...")
    print("=" * 70)
    
    all_questions = generator.generate_rap_calibrated_batch(
        questions_per_level=10,
        include_risk_seeking=True,
        include_risk_averse=True
    )
    output_path = Path("sample_questions.json")
    save_questions_json(all_questions, "sample_questions.json")
    save_questions_jsonl(all_questions, "sample_questions.jsonl")
    
    print(f"\nSaved {len(all_questions)} questions to:")
