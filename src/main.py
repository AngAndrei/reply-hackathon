from langfuse import Langfuse

from src.config.config import config
from src.tracing.langfuse_helpers import create_model, generate_session_id, run_llm_call

DEFAULT_QUESTIONS = [
    "What is machine learning?",
    "Explain neural networks briefly.",
    "What is the difference between AI and ML?",
]


def run_demo(questions: list[str] | None = None) -> str:
    questions = questions or DEFAULT_QUESTIONS
    langfuse_client = Langfuse(
        public_key=config.LANGFUSE_PUBLIC_KEY,
        secret_key=config.LANGFUSE_SECRET_KEY,
        host=config.LANGFUSE_HOST,
    )
    model = create_model()
    session_id = generate_session_id()

    print(f"Model configured: {config.MODEL_ID}")
    print(f"Session ID: {session_id}")
    print(f"Making {len(questions)} agent calls with Langfuse tracing...\n")

    try:
        for index, question in enumerate(questions, 1):
            response = run_llm_call(session_id, model, question)
            print(f"Call {index}: {question}")
            print(f"  Response: {response[:160]}...\n")
    finally:
        langfuse_client.flush()

    print("All traces flushed to Langfuse.")
    print(f"Grouped under session: {session_id}")
    return session_id


def main() -> None:
    run_demo()


if __name__ == "__main__":
    main()
