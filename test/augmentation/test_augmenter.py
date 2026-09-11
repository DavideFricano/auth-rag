from auth_rag.augmentation.augmenter import PromptAugmenter

aug = PromptAugmenter(system="Sei un medico.")


def test_system_prompt_contains_text():
    result = aug.build_system_prompt("Sei un medico.")
    assert "SISTEMA:" in result
    assert "Sei un medico." in result


def test_query_prompt_contains_text():
    result = aug.build_query_prompt("Qual è la diagnosi?")
    assert "DOMANDA:" in result
    assert "Qual è la diagnosi?" in result


def test_context_prompt_contains_text():
    result = aug.build_context_prompt("Il paziente ha febbre.")
    assert "CONTESTO:" in result
    assert "Il paziente ha febbre." in result


def test_context_prompt_empty_shows_no_source():
    assert "Nessuna fonte trovata" in aug.build_context_prompt("")


def test_build_returns_system_and_user_messages():
    messages = aug.build("Qual è la diagnosi?", "Il paziente ha febbre.")
    assert [m.role for m in messages] == ["system", "user"]
    assert "Sei un medico." in messages[0].content
    assert "Qual è la diagnosi?" in messages[1].content
    assert "Il paziente ha febbre." in messages[1].content
