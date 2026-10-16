from app.services.text_sanitize import strip_nul


def test_removes_nul_characters():
    assert strip_nul("Kubernetes\x00 networking\x00") == "Kubernetes networking"


def test_leaves_clean_text_untouched():
    text = "SR-IOV on Intel NICs — résumé ✓\n\ttabs and newlines stay"
    assert strip_nul(text) == text


def test_empty_string():
    assert strip_nul("") == ""


def test_result_is_json_safe_for_postgres():
    # Postgres JSON/JSONB rejects the \u0000 escape that json.dumps emits for a NUL.
    import json
    assert "\\u0000" not in json.dumps(strip_nul("a\x00b"))
