import math

import pytest

from src.generation.trustmargin import Likelihoods, arbitrate


def test_formula_and_strict_threshold():
    memory = Likelihoods(-1, -2, -3)
    rag = Likelihoods(-3, -1, -4)
    # prior=-2; binding=(3)-(1)=2; M=-1
    result = arbitrate(memory, rag)
    assert result["prior_margin"] == -2
    assert result["binding_margin"] == 2
    assert result["margin"] == -1
    assert result["selected"] == "rag"
    assert arbitrate(memory, rag, tau=-1)["selected"] == "d2l"
    assert arbitrate(memory, rag, lambda_bind=0)["selected"] == "d2l"


def test_passage_salience_is_subtracted():
    memory = Likelihoods(-1, -2, -3)
    rag = Likelihoods(-2, -1, -1)
    result = arbitrate(memory, rag)
    assert result["binding_margin"] == -1
    assert result["selected"] == "d2l"


@pytest.mark.parametrize("value", [math.nan, math.inf, -math.inf])
def test_nonfinite_scores_rejected(value):
    with pytest.raises(ValueError):
        Likelihoods(value, -1, -2)


@pytest.mark.parametrize("kwargs", [{"lambda_bind": -1}, {"tau": math.nan}])
def test_invalid_parameters_rejected(kwargs):
    with pytest.raises(ValueError):
        arbitrate(Likelihoods(-1, -1, -1), Likelihoods(-2, -2, -2), **kwargs)
