"""エージェントの配線テスト。

TestModel は実際のLLMを呼ばない偽モデルで、登録されたツールを全部呼んでから
ダミーの回答を返す。APIキーも利用枠も使わずに、
「ツールが登録され、呼ばれ、結果がモデルに戻るか」を確かめられる。
"""

from types import SimpleNamespace

import pytest
from pydantic_ai import DeferredToolRequests, ToolApproved, ToolDenied
from pydantic_ai.messages import ToolCallPart, ToolReturnPart
from pydantic_ai.models.test import TestModel

from app.agent import agent, run_with_approval

BASE_TOOLS = {"search_glossary", "get_lecture_summary"}


def _parts(result, kind):
    return [p for m in result.all_messages() for p in m.parts if isinstance(p, kind)]


def test_base_tools_are_registered_and_called():
    model = TestModel()
    with agent.override(model=model):
        result = agent.run_sync("LoRAとは？")

    offered = {t.name for t in model.last_model_request_parameters.function_tools}
    assert BASE_TOOLS <= offered  # 演習でツールを足しても壊れないよう「含む」で判定

    called = {p.tool_name for p in _parts(result, ToolCallPart)}
    assert BASE_TOOLS <= called
    # 承認が必要なツール（requires_approval=True）は、実行されずに承認待ちになるので戻り値が無い
    waiting = len(result.output.approvals) if isinstance(result.output, DeferredToolRequests) else 0
    assert len(_parts(result, ToolReturnPart)) + waiting == len(_parts(result, ToolCallPart))
    assert result.output


def test_glossary_result_reaches_the_model():
    model = TestModel(call_tools=["search_glossary"])
    with agent.override(model=model):
        result = agent.run_sync("RAGとは？")
    returns = _parts(result, ToolReturnPart)
    assert returns and returns[0].tool_name == "search_glossary"


@pytest.mark.parametrize(
    ("answer", "expected_type"),
    [("y", ToolApproved), ("n", ToolDenied)],
)
def test_deferred_tool_approval_is_requested_and_resumed(
    monkeypatch, capsys, answer, expected_type
):
    call = ToolCallPart(
        tool_name="save_learning_log",
        args={"learning_log": "今日学んだこと"},
        tool_call_id="call-1",
    )
    pending = SimpleNamespace(
        output=DeferredToolRequests(approvals=[call]),
        all_messages=lambda: ["conversation history"],
    )
    completed = SimpleNamespace(output="完了")

    class FakeAgent:
        def __init__(self):
            self.calls = []
            self.results = [pending, completed]

        def run_sync(self, *args, **kwargs):
            self.calls.append((args, kwargs))
            return self.results.pop(0)

    fake_agent = FakeAgent()
    monkeypatch.setattr("builtins.input", lambda _prompt: answer)

    result = run_with_approval(fake_agent, "学習記録を保存して", model=object())

    assert result is completed
    assert "今日学んだこと" in capsys.readouterr().out
    resume_kwargs = fake_agent.calls[1][1]
    assert resume_kwargs["message_history"] == ["conversation history"]
    approval = resume_kwargs["deferred_tool_results"].approvals["call-1"]
    assert isinstance(approval, expected_type)
