from types import SimpleNamespace

from app import ask_many_times as ask_module


def test_main_reuses_chat_for_follow_up_questions(monkeypatch, capsys):
    sent_messages = []

    class FakeChat:
        def send_message(self, message):
            sent_messages.append(message)
            return SimpleNamespace(
                usage_metadata=SimpleNamespace(
                    prompt_token_count=12,
                    candidates_token_count=8,
                ),
                text=f"answer to {message}",
            )

    class FakeChats:
        def create(self, **kwargs):
            assert kwargs == {"model": ask_module.MODEL}
            return FakeChat()

    class FakeClient:
        chats = FakeChats()

    def fake_client(*, api_key):
        assert api_key == "test-key"
        return FakeClient()

    def fake_input(prompt):
        assert prompt == "[質問] "
        return next(questions)

    questions = iter(["first question", "follow-up", "終了"])
    monkeypatch.setattr(ask_module.genai, "Client", fake_client)
    monkeypatch.setattr(ask_module, "get_api_key", lambda: "test-key")
    monkeypatch.setattr("builtins.input", fake_input)

    ask_module.main()

    output = capsys.readouterr().out
    assert sent_messages == ["first question", "follow-up"]
    assert "answer to first question" in output
    assert "answer to follow-up" in output
    assert output.count("[トークン] 入力: 12, 出力: 8") == 2
