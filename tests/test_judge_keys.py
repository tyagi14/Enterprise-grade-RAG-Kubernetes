import pytest

import evals.metrics as M

ENV_NAMES = [
    "JUDGE_GROQ", "JUDGE_GROQ_API_KEY", "GROQ_API_KEY",
    *[f"JUDGE_GROQ_{n}" for n in range(2, 7)], *[f"JUDGE_GROQ_API_KEY_{n}" for n in range(2, 7)],
]


@pytest.fixture(autouse=True)
def clean_env(monkeypatch):
    for name in ENV_NAMES:
        monkeypatch.delenv(name, raising=False)


def test_single_key(monkeypatch):
    monkeypatch.setenv("JUDGE_GROQ", "k1")
    assert M._judge_keys() == ("k1",)


def test_numbered_keys_are_collected_in_order(monkeypatch):
    monkeypatch.setenv("JUDGE_GROQ", "k1")
    monkeypatch.setenv("JUDGE_GROQ_2", "k2")
    monkeypatch.setenv("JUDGE_GROQ_3", "k3")
    assert M._judge_keys() == ("k1", "k2", "k3")


def test_gap_in_numbering_is_tolerated(monkeypatch):
    monkeypatch.setenv("JUDGE_GROQ", "k1")
    monkeypatch.setenv("JUDGE_GROQ_3", "k3")
    assert M._judge_keys() == ("k1", "k3")


def test_duplicate_keys_are_dropped(monkeypatch):
    monkeypatch.setenv("JUDGE_GROQ", "same")
    monkeypatch.setenv("JUDGE_GROQ_2", "same")
    assert M._judge_keys() == ("same",)


def test_api_key_style_names_work(monkeypatch):
    monkeypatch.setenv("JUDGE_GROQ_API_KEY", "k1")
    monkeypatch.setenv("JUDGE_GROQ_API_KEY_2", "k2")
    assert M._judge_keys() == ("k1", "k2")


def test_no_key_is_an_error():
    with pytest.raises(ValueError):
        M._judge_keys()


def test_production_key_is_never_used_as_a_judge_key(monkeypatch):
    # Eval runs must not spend the production Groq organisation's daily budget.
    monkeypatch.setenv("GROQ_API_KEY", "production-key")
    with pytest.raises(ValueError):
        M._judge_keys()


def test_production_key_is_ignored_when_judge_keys_exist(monkeypatch):
    monkeypatch.setenv("GROQ_API_KEY", "production-key")
    monkeypatch.setenv("JUDGE_GROQ", "judge-key")
    assert M._judge_keys() == ("judge-key",)
