"""会話履歴を保ちながら、LLMに続けて質問するスクリプト。

使い方:
    uv run python -m app.ask_many_times

空の質問、exit、quit、終了のいずれかで終了する。
"""

from google import genai
from google.genai import errors

from app.config import MODEL, api_error_message, get_api_key


def main() -> None:
    client = genai.Client(api_key=get_api_key())
    chat = client.chats.create(model=MODEL)
    print(f"[model] {MODEL}")
    print("質問を入力してください（終了するには空入力または「終了」）。")

    try:
        while True:
            question = input("[質問] ")
            if question.strip().lower() in {"", "exit", "quit", "終了"}:
                break

            try:
                response = chat.send_message(question)
            except errors.APIError as e:
                print(api_error_message(e.code, e.message))
                raise SystemExit(1) from e

            usage = response.usage_metadata
            if usage is not None:
                print(
                    f"[トークン] 入力: {usage.prompt_token_count}, "
                    f"出力: {usage.candidates_token_count}"
                )
            print(response.text)
    except (EOFError, KeyboardInterrupt):
        print()


if __name__ == "__main__":
    main()
