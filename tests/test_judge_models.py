from types import SimpleNamespace

from evals.judge_models import JudgeModelChain, KeyHolder, build_steps, is_daily_limit, max_tokens_for, parse_chain, reasoning_effort_for


def _llm(model="qwen/qwen3.8-27b"):
    return SimpleNamespace(model=model, model_args={"max_tokens": 4096, "reasoning_effort": "none"})


def test_reasoning_effort_depends_on_model_family():
    assert reasoning_effort_for("openai/gpt-oss-20b") == "low"
    assert reasoning_effort_for("qwen/qwen3.8-27b") == "none"


def test_reasoning_effort_env_override(monkeypatch):
    monkeypatch.setenv("JUDGE_REASONING_EFFORT", "high")
    assert reasoning_effort_for("qwen/qwen3.8-27b") == "high"


def test_parse_chain_default_excludes_production_main_model():
    chain = parse_chain(models_env="", model_env="")
    assert chain[0] == "qwen/qwen3.8-27b"
    assert "openai/gpt-oss-120b" not in chain


def test_parse_chain_explicit_list_wins():
    assert parse_chain(models_env=" a/x , b/y ", model_env="c/z") == ("a/x", "b/y")


def test_parse_chain_single_model_env_goes_first_then_defaults():
    chain = parse_chain(models_env="", model_env="openai/gpt-oss-20b")
    assert chain[0] == "openai/gpt-oss-20b"
    assert len(chain) == len(set(chain))


def test_max_tokens_is_roomy_for_reasoning_models_and_small_for_the_rest(monkeypatch):
    monkeypatch.delenv("JUDGE_MAX_TOKENS", raising=False)
    assert max_tokens_for("openai/gpt-oss-20b") == 4096
    assert max_tokens_for("qwen/qwen3.8-27b") == 2048


def test_max_tokens_env_override(monkeypatch):
    monkeypatch.setenv("JUDGE_MAX_TOKENS", "1500")
    assert max_tokens_for("openai/gpt-oss-20b") == 1500


def test_advance_updates_max_tokens_for_the_new_model():
    llm = _llm()
    chain = JudgeModelChain(llm, build_steps(("qwen/qwen3.8-27b", "openai/gpt-oss-20b"), 1), KeyHolder(("k",)))
    chain.advance()
    assert llm.model_args["max_tokens"] == 4096


def test_build_steps_exhausts_both_keys_on_a_model_before_changing_model():
    steps = build_steps(("m/a", "m/b"), key_count=2)
    assert [(st.model, st.key_index) for st in steps] == [("m/a", 0), ("m/a", 1), ("m/b", 0), ("m/b", 1)]


def test_build_steps_single_key():
    assert [(st.model, st.key_index) for st in build_steps(("m/a", "m/b"), key_count=1)] == [("m/a", 0), ("m/b", 0)]


def test_advance_switches_model_and_reasoning_effort_without_mutating_old_args():
    llm = _llm()
    old_args = llm.model_args
    chain = JudgeModelChain(llm, build_steps(("qwen/qwen3.8-27b", "openai/gpt-oss-20b"), 1), KeyHolder(("k1",)))
    assert chain.advance() == "openai/gpt-oss-20b"
    assert llm.model == "openai/gpt-oss-20b"
    assert llm.model_args["reasoning_effort"] == "low"
    assert old_args["reasoning_effort"] == "none"


def test_advance_to_second_key_keeps_model_and_swaps_key():
    llm = _llm()
    holder = KeyHolder(("key-one", "key-two"))
    chain = JudgeModelChain(llm, build_steps(("qwen/qwen3.8-27b",), 2), holder)
    assert holder.key == "key-one"
    assert chain.advance() == "qwen/qwen3.8-27b"
    assert holder.key == "key-two"
    assert llm.model == "qwen/qwen3.8-27b"


def test_advance_returns_none_when_chain_is_exhausted():
    chain = JudgeModelChain(_llm(), build_steps(("only/one",), 1), KeyHolder(("k",)))
    assert chain.advance() is None


def test_daily_limit_detection():
    assert is_daily_limit("Rate limit reached for model x on tokens per day (TPD): Limit 200000")
    assert not is_daily_limit("Rate limit reached on tokens per minute (TPM)")
    assert not is_daily_limit("json_validate_failed")
