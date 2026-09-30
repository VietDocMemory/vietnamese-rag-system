"""TrustMargin, equations (3)-(10), arXiv:2606.08397v1.

The memory candidate here is D2L rather than the paper's unadapted Direct answer.
All six likelihoods must be evaluated with one fixed model/adapter state.
"""

from dataclasses import asdict, dataclass
import math


@dataclass(frozen=True)
class Likelihoods:
    question: float
    question_context: float
    context: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in asdict(self).values()):
            raise ValueError("TrustMargin requires finite, mean token log-likelihoods.")


def arbitrate(memory: Likelihoods, rag: Likelihoods, lambda_bind=0.5, tau=-1.5) -> dict:
    if not math.isfinite(lambda_bind) or lambda_bind < 0 or not math.isfinite(tau):
        raise ValueError("Invalid TrustMargin parameters.")
    prior = rag.question - memory.question
    binding = (rag.question_context - rag.context) - (memory.question_context - memory.context)
    margin = prior + lambda_bind * binding
    return {
        "selected": "rag" if margin > tau else "d2l",
        "prior_margin": prior,
        "binding_margin": binding,
        "margin": margin,
        "lambda_bind": lambda_bind,
        "tau": tau,
        "likelihoods": {"d2l": asdict(memory), "rag": asdict(rag)},
    }
