import ulid
import importlib
from langchain_core.messages import HumanMessage
from langchain_openai import ChatOpenAI
from langchain_core.language_models.fake_chat_models import FakeListChatModel

from src.config.config import config

ChatOllama = None
try:
    ollama_module = importlib.import_module("langchain_ollama")
    ChatOllama = getattr(ollama_module, "ChatOllama", None)
except ImportError:
    ChatOllama = None

OLLAMA_NATIVE_AVAILABLE = ChatOllama is not None

try:
    from langfuse import observe
    from langfuse.langchain import CallbackHandler
    LANGFUSE_AVAILABLE = True
except ImportError:
    LANGFUSE_AVAILABLE = False

def generate_session_id() -> str:
    if config.LOCAL_DEV_NO_TRACING:
        return f"local-test-{ulid.new().str}"
        
    team = config.TEAM_NAME.strip().replace(" ", "-") or "Geese"
    return f"{team}-{ulid.new().str}"

def create_model(model_id: str = None, temperature: float = None):
    target_model = model_id or config.MODEL_ID
    target_temp = temperature if temperature is not None else config.MODEL_TEMPERATURE

    if config.USE_MOCK_LLM:
        return FakeListChatModel(responses=["""{"verdict": "fraud", "confidence": 90, "reasoning": ["Mock"]}"""])
        
    if config.USE_LOCAL_LLM:
        print(f"🤖 [LOCAL AI] Booting up {target_model}...")
        local_url = config.LOCAL_LLM_URL.rstrip("/")
        # Native Ollama APIs are served on /api/*; strip optional /v1 for compatibility.
        if local_url.endswith("/v1"):
            local_url = local_url[:-3]

        if OLLAMA_NATIVE_AVAILABLE:
            return ChatOllama(
                base_url=local_url,
                model=target_model,
                temperature=target_temp,
            )

        openai_compatible_url = config.LOCAL_LLM_URL
        return ChatOpenAI(
            api_key="local-dev-key",
            base_url=openai_compatible_url,
            model=target_model,
            temperature=target_temp,
        )
        
    return ChatOpenAI(
        api_key=config.OPENROUTER_API_KEY,
        base_url="https://openrouter.ai/api/v1",
        model=target_model,
        temperature=target_temp,
    )

def _invoke_langchain_traced(
    model: ChatOpenAI,
    prompt: str,
    session_id: str,
) -> str:
    langfuse_handler = CallbackHandler()
    response = model.invoke(
        [HumanMessage(content=prompt)],
        config={
            "callbacks": [langfuse_handler],
            "metadata": {"langfuse_session_id": session_id},
        },
    )
    return str(response.content)

def _invoke_langchain_local(model: ChatOpenAI, prompt: str) -> str:
    response = model.invoke([HumanMessage(content=prompt)])
    return str(response.content)

def run_llm_call(session_id: str, model: ChatOpenAI, prompt: str) -> str:
    if config.LOCAL_DEV_NO_TRACING or not LANGFUSE_AVAILABLE:
        print(f"[LOCAL MODE] Running untraced LLM call for session: {session_id}")
        return _invoke_langchain_local(model, prompt)
    
    @observe()
    def _traced_call():
        return _invoke_langchain_traced(model, prompt, session_id)
        
    return _traced_call()
