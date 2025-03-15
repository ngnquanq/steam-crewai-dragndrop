
import streamlit as st
from crewai import Agent, Task, Crew, Process
from langchain_openai import ChatOpenAI
from langchain_groq import ChatGroq
from langchain_anthropic import ChatAnthropic
from dotenv import load_dotenv
import os
from crewai_tools import *





load_dotenv()

def create_lmstudio_llm(model, temperature):
    api_base = os.getenv('LMSTUDIO_API_BASE')
    os.environ["OPENAI_API_KEY"] = "lm-studio"
    os.environ["OPENAI_API_BASE"] = api_base
    if api_base:
        return ChatOpenAI(openai_api_key='lm-studio', openai_api_base=api_base, temperature=temperature)
    else:
        raise ValueError("LM Studio API base not set in .env file")

def create_openai_llm(model, temperature):
    safe_pop_env_var('OPENAI_API_KEY')
    safe_pop_env_var('OPENAI_API_BASE')
    load_dotenv(override=True)
    api_key = os.getenv('OPENAI_API_KEY')
    api_base = os.getenv('OPENAI_API_BASE', 'https://api.openai.com/v1/')
    if api_key:
        return ChatOpenAI(openai_api_key=api_key, openai_api_base=api_base, model_name=model, temperature=temperature)
    else:
        raise ValueError("OpenAI API key not set in .env file")

def create_groq_llm(model, temperature):
    api_key = os.getenv('GROQ_API_KEY')
    if api_key:
        return ChatGroq(groq_api_key=api_key, model_name=model, temperature=temperature)
    else:
        raise ValueError("Groq API key not set in .env file")

def create_anthropic_llm(model, temperature):
    api_key = os.getenv('ANTHROPIC_API_KEY')
    if api_key:
        return ChatAnthropic(anthropic_api_key=api_key, model_name=model, temperature=temperature)
    else:
        raise ValueError("Anthropic API key not set in .env file")

def safe_pop_env_var(key):
    try:
        os.environ.pop(key)
    except KeyError:
        pass
        
LLM_CONFIG = {
    "OpenAI": {
        "create_llm": create_openai_llm
    },
    "Groq": {
        "create_llm": create_groq_llm
    },
    "LM Studio": {
        "create_llm": create_lmstudio_llm
    },
    "Anthropic": {
        "create_llm": create_anthropic_llm
    }
}

def create_llm(provider_and_model, temperature=0.1):
    provider, model = provider_and_model.split(": ")
    create_llm_func = LLM_CONFIG.get(provider, {}).get("create_llm")
    if create_llm_func:
        return create_llm_func(model, temperature)
    else:
        raise ValueError(f"LLM provider {provider} is not recognized or not supported")

def load_agents():
    agents = [
        
Agent(
    role="Information Extraction Agent",
    backstory="B\u1ea1n l\u00e0 hi\u1ec7u tr\u01b0\u1edfng m\u1ed9t tr\u01b0\u1eddng THPT, \u0111ang c\u1ea7n tr\u00edch xu\u1ea5t th\u00f4ng tin t\u1eeb hai t\u00e0i li\u1ec7u sau:\n\n    BC TK C\u00d4NG T\u00c1C C\u00d4NG \u0110O\u00c0N N\u0102M 2023-2024\n        B\u00e1o c\u00e1o t\u1ed5ng k\u1ebft ho\u1ea1t \u0111\u1ed9ng c\u00f4ng \u0111o\u00e0n, k\u1ebft qu\u1ea3 c\u00e1c ch\u01b0\u01a1ng tr\u00ecnh h\u1ed7 tr\u1ee3 gi\u00e1o vi\u00ean.\n        C\u00e1c kh\u00f3 kh\u0103n, th\u00e1ch th\u1ee9c v\u00e0 b\u00e0i h\u1ecdc r\u00fat ra.\n\n    KH S\u1ed0 01 - C\u00f4ng t\u00e1c c\u1ee5m TT-00 n\u0103m h\u1ecdc 2024-2025\n        K\u1ebf ho\u1ea1ch t\u1ed5 ch\u1ee9c ho\u1ea1t \u0111\u1ed9ng c\u1ee7a c\u1ee5m tr\u01b0\u1eddng, \u0111\u1ecbnh h\u01b0\u1edbng \u0111\u1ed5i m\u1edbi, h\u1ee3p t\u00e1c.\n        Danh s\u00e1ch c\u00e1c ho\u1ea1t \u0111\u1ed9ng chi\u1ebfn l\u01b0\u1ee3c, l\u1ed9 tr\u00ecnh th\u1ef1c hi\u1ec7n.",
    goal="M\u1ee4C TI\u00caU (Goal)\n\n    Tr\u00edch xu\u1ea5t th\u00f4ng tin v\u1ec1:\n        C\u00e1c ho\u1ea1t \u0111\u1ed9ng th\u00e0nh c\u00f4ng ho\u1eb7c ch\u01b0a \u0111\u1ea1t k\u1ebft qu\u1ea3 t\u1ed1t trong n\u0103m tr\u01b0\u1edbc.\n        Nh\u1eefng \u0111\u1ec1 xu\u1ea5t, n\u1ed9i dung m\u1edbi trong k\u1ebf ho\u1ea1ch c\u1ee5m tr\u01b0\u1eddng cho n\u0103m h\u1ecdc 2024-2025.\n\nY\u00caU C\u1ea6U AI PH\u1ea2I TU\u00c2N THEO (Instruction Steps)\n\n    T\u00f3m t\u1eaft & Tr\u00edch xu\u1ea5t th\u00f4ng tin ch\u00ednh (Summarize & Extract)\n        T\u1eadp trung li\u1ec7t k\u00ea v\u00e0 t\u00f3m l\u01b0\u1ee3c chi ti\u1ebft t\u1eeb hai t\u00e0i li\u1ec7u.\n        N\u1ed9i dung c\u1ea7n ch\u00ednh x\u00e1c, n\u00eau r\u00f5 nh\u1eefng ho\u1ea1t \u0111\u1ed9ng, k\u1ebft qu\u1ea3 v\u00e0 \u0111\u1ec1 xu\u1ea5t quan tr\u1ecdng.\n\n    L\u01b0u \u00fd: Hi\u1ec7n t\u1ea1i kh\u00f4ng c\u1ea7n ph\u00e2n t\u00edch s\u00e2u, x\u00e2y d\u1ef1ng k\u1ebf ho\u1ea1ch th\u00ed \u0111i\u1ec3m hay ti\u00eau ch\u00ed \u0111\u00e1nh gi\u00e1. M\u1ee5c ti\u00eau duy nh\u1ea5t l\u00e0 tr\u00edch xu\u1ea5t th\u00f4ng tin m\u1ed9t c\u00e1ch \u0111\u1ea7y \u0111\u1ee7 v\u00e0 ch\u00ednh x\u00e1c t\u1eeb hai t\u00e0i li\u1ec7u tr\u00ean.",
    allow_delegation=True,
    verbose=True,
    tools=[DOCXSearchTool(), PDFSearchTool(), DOCXSearchTool(), TXTSearchTool()],
    llm=create_llm("OpenAI: gpt-4o-mini", 0.05)
)
            ,
        
Agent(
    role="Idea Brainstorming Agent",
    backstory="B\u1ed0I C\u1ea2NH (Context)\nB\u1ea1n l\u00e0 hi\u1ec7u tr\u01b0\u1edfng m\u1ed9t tr\u01b0\u1eddng THPT, c\u1ea7n tham kh\u1ea3o v\u00e0 brainstorm \u1edf m\u1ee9c \u0111\u1ed9 t\u1ed5ng qu\u00e1t (high-level) v\u1ec1 c\u00e1c \u00fd t\u01b0\u1edfng m\u1edbi, \u00e1p d\u1ee5ng t\u1eeb K\u1ebf ho\u1ea1ch c\u1ee7a B\u1ed9 v\u00e0 S\u1edf cho n\u0103m h\u1ecdc 2024-2025.\n",
    goal="NHI\u1ec6M V\u1ee4 (Task)\n\n    H\u00e3y \u0111\u1ec1 xu\u1ea5t nh\u1eefng \u00fd t\u01b0\u1edfng, g\u1ee3i \u00fd \u00e1p d\u1ee5ng trong k\u1ebf ho\u1ea1ch d\u1ef1a tr\u00ean:\n        K\u1ebft qu\u1ea3 c\u1ee7a c\u00e1c tr\u01b0\u1eddng kh\u00e1c (trong v\u00e0 ngo\u00e0i n\u01b0\u1edbc)\n        C\u00e1c nghi\u00ean c\u1ee9u, m\u00f4 h\u00ecnh gi\u00e1o d\u1ee5c hi\u1ec7u qu\u1ea3\n        Th\u1ef1c t\u1ebf c\u1ee7a tr\u01b0\u1eddng ch\u00fang ta (ngu\u1ed3n l\u1ef1c, quy m\u00f4, m\u1ee5c ti\u00eau)\n    M\u1ee5c ti\u00eau l\u00e0 t\u1ea1o ra nhi\u1ec1u ph\u01b0\u01a1ng \u00e1n s\u00e1ng t\u1ea1o, c\u00f3 t\u00ednh kh\u1ea3 thi, ph\u00f9 h\u1ee3p v\u1edbi xu h\u01b0\u1edbng gi\u00e1o d\u1ee5c hi\u1ec7n \u0111\u1ea1i.\n\nH\u01af\u1edaNG D\u1eaaN AI (Instruction Steps)\n\n    Ph\u00e2n t\u00edch t\u1ed5ng th\u1ec3: D\u1ef1a tr\u00ean c\u00e1c ch\u1ec9 \u0111\u1ea1o ho\u1eb7c \u0111\u1ecbnh h\u01b0\u1edbng c\u1ee7a K\u1ebf ho\u1ea1ch B\u1ed9/S\u1edf.\n    K\u1ebft h\u1ee3p k\u1ebft qu\u1ea3 & kinh nghi\u1ec7m: Tham chi\u1ebfu m\u00f4 h\u00ecnh th\u00e0nh c\u00f4ng t\u1ea1i c\u00e1c tr\u01b0\u1eddng/th\u00e0nh ph\u1ed1/n\u01b0\u1edbc kh\u00e1c.\n    \u0110\u1ec1 xu\u1ea5t \u00fd t\u01b0\u1edfng m\u1edbi:\n        Suy ngh\u0129 \u0111a chi\u1ec1u (c\u00f3 th\u1ec3 c\u00f3 nhi\u1ec1u phi\u00ean b\u1ea3n, k\u1ecbch b\u1ea3n kh\u00e1c nhau).\n        Ch\u00fa tr\u1ecdng \u0111\u1ed5i m\u1edbi ph\u01b0\u01a1ng ph\u00e1p d\u1ea1y v\u00e0 h\u1ecdc, h\u1ee3p t\u00e1c li\u00ean tr\u01b0\u1eddng, \u1ee9ng d\u1ee5ng c\u00f4ng ngh\u1ec7.\n    \u0110\u00e1nh gi\u00e1 s\u01a1 b\u1ed9:\n        T\u00ednh ph\u00f9 h\u1ee3p v\u1edbi b\u1ed1i c\u1ea3nh tr\u01b0\u1eddng.\n        T\u00ednh kh\u1ea3 thi, nh\u1eefng \u0111i\u1ec1u ki\u1ec7n ti\u00ean quy\u1ebft c\u1ea7n c\u00f3.\n    T\u1ed5ng h\u1ee3p: Li\u1ec7t k\u00ea c\u00e1c \u00fd t\u01b0\u1edfng cu\u1ed1i c\u00f9ng \u1edf d\u1ea1ng danh s\u00e1ch g\u1ecdn g\u00e0ng, d\u1ec5 hi\u1ec3u.\n\n    L\u01b0u \u00fd: \u0110\u00e2y l\u00e0 b\u01b0\u1edbc \u201cbrainstorm high-level\u201d, n\u00ean khuy\u1ebfn kh\u00edch m\u1ecdi \u00fd ki\u1ebfn m\u1edbi l\u1ea1, s\u00e1ng t\u1ea1o. Ch\u01b0a c\u1ea7n ph\u00e2n t\u00edch qu\u00e1 chi ti\u1ebft v\u1ec1 ng\u00e2n s\u00e1ch hay nh\u00e2n s\u1ef1, ch\u1ec9 c\u1ea7n \u0111\u1ea3m b\u1ea3o m\u1ed7i \u00fd t\u01b0\u1edfng c\u00f3 \u0111\u1ecbnh h\u01b0\u1edbng, m\u1ee5c ti\u00eau r\u00f5 r\u00e0ng.",
    allow_delegation=True,
    verbose=True,
    tools=[],
    llm=create_llm("OpenAI: gpt-4o-mini", 0.05)
)
            ,
        
Agent(
    role="Pilot Planning Agent",
    backstory="B\u1ea1n l\u00e0 hi\u1ec7u tr\u01b0\u1edfng m\u1ed9t tr\u01b0\u1eddng THPT, v\u1eeba ph\u00e2n t\u00edch b\u00e1o c\u00e1o c\u00f4ng t\u00e1c c\u00f4ng \u0111o\u00e0n, brainstorm \u00fd t\u01b0\u1edfng \u00e1p d\u1ee5ng t\u1eeb K\u1ebf ho\u1ea1ch B\u1ed9/S\u1edf, v\u00e0 n\u00eau ra nhi\u1ec1u \u0111\u1ec1 xu\u1ea5t c\u1ea3i thi\u1ec7n. B\u00e2y gi\u1edd, b\u1ea1n c\u1ea7n x\u00e2y d\u1ef1ng K\u1ebf ho\u1ea1ch th\u00ed \u0111i\u1ec3m (Pilot Plan) c\u1ee5 th\u1ec3, d\u1ef1a tr\u00ean:\n    C\u00e1c \u0111\u1ec1 xu\u1ea5t c\u1ea3i thi\u1ec7n t\u1eeb b\u00e1o c\u00e1o c\u00f4ng \u0111o\u00e0n.\n    C\u00e1c \u00fd t\u01b0\u1edfng brainstorm \u0111\u00e3 c\u00f3.\n    C\u00e2n nh\u1eafc nh\u1eefng m\u1eb7t l\u1ee3i \u2013 m\u1eb7t h\u1ea1i, v\u00e0 t\u00ednh kh\u1ea3 thi trong th\u1ef1c t\u1ebf nh\u00e0 tr\u01b0\u1eddng.",
    goal="NHI\u1ec6M V\u1ee4 (Task)\nH\u00e3y \u0111\u1ec1 xu\u1ea5t m\u1ed9t K\u1ebf ho\u1ea1ch th\u00ed \u0111i\u1ec3m v\u1edbi c\u00e1c b\u01b0\u1edbc tri\u1ec3n khai c\u1ee5 th\u1ec3, c\u00f3 l\u1ecbch tr\u00ecnh r\u00f5 r\u00e0ng, th\u1ec3 hi\u1ec7n:\n\n    M\u1ee5c ti\u00eau: N\u00eau r\u00f5 m\u1ee5c \u0111\u00edch mong mu\u1ed1n \u0111\u1ea1t \u0111\u01b0\u1ee3c.\n    Ph\u1ea1m vi \u00e1p d\u1ee5ng: Ph\u1ea1m vi, \u0111\u1ed1i t\u01b0\u1ee3ng th\u00ed \u0111i\u1ec3m (kh\u1ed1i l\u1edbp, nh\u00f3m gi\u00e1o vi\u00ean\u2026).\n    C\u00e1c b\u01b0\u1edbc tri\u1ec3n khai:\n        Th\u1eddi gian, ngu\u1ed3n l\u1ef1c (nh\u00e2n l\u1ef1c, t\u00e0i ch\u00ednh) c\u1ea7n thi\u1ebft.\n        C\u00e1ch th\u1ee9c th\u1ef1c hi\u1ec7n: \u0111\u00e0o t\u1ea1o, h\u1ed9i th\u1ea3o, ho\u1ea1t \u0111\u1ed9ng ngo\u1ea1i kh\u00f3a\u2026\n    Ti\u00eau ch\u00ed \u0111\u00e1nh gi\u00e1 th\u00e0nh c\u00f4ng: (ch\u1ec9 n\u00eau s\u01a1 b\u1ed9, ch\u01b0a c\u1ea7n chi ti\u1ebft).\n    Qu\u1ea3n l\u00fd r\u1ee7i ro & ki\u1ec3m so\u00e1t: Nh\u1eefng r\u1ee7i ro c\u00f3 th\u1ec3 x\u1ea3y ra, c\u00e1ch gi\u1ea3m thi\u1ec3u.\n\nH\u01af\u1edaNG D\u1eaaN AI (Instruction Steps)\n\n    T\u1ed5ng h\u1ee3p \u0111\u1ec1 xu\u1ea5t c\u1ea3i thi\u1ec7n: T\u1eeb b\u01b0\u1edbc tr\u01b0\u1edbc, x\u00e1c \u0111\u1ecbnh nh\u1eefng gi\u1ea3i ph\u00e1p \u01b0u ti\u00ean.\n    Ch\u1ecdn \u00fd t\u01b0\u1edfng ph\u00f9 h\u1ee3p: Trong s\u1ed1 c\u00e1c \u00fd t\u01b0\u1edfng brainstorm, ch\u1ecdn l\u1ecdc nh\u1eefng \u00fd ki\u1ebfn c\u00f3 t\u00ednh kh\u1ea3 thi cao, g\u1eafn li\u1ec1n v\u1edbi t\u00ecnh h\u00ecnh nh\u00e0 tr\u01b0\u1eddng.\n    X\u00e2y d\u1ef1ng k\u1ebf ho\u1ea1ch th\u00ed \u0111i\u1ec3m:\n        T\u1eebng b\u01b0\u1edbc r\u00f5 r\u00e0ng, th\u1ef1c t\u1ebf.\n        C\u00e2n nh\u1eafc k\u1ef9 m\u1eb7t l\u1ee3i \u2013 m\u1eb7t h\u1ea1i, v\u00e0 \u0111i\u1ec1u ch\u1ec9nh k\u1ebf ho\u1ea1ch \u0111\u1ec3 h\u1ea1n ch\u1ebf r\u1ee7i ro.\n    Gi\u1ea3i th\u00edch t\u00ednh kh\u1ea3 thi: Ch\u1ee9ng minh t\u1ea1i sao k\u1ebf ho\u1ea1ch c\u00f3 th\u1ec3 th\u1ef1c hi\u1ec7n \u0111\u01b0\u1ee3c, n\u00eau c\u00e1c ngu\u1ed3n l\u1ef1c \u0111\u00e1p \u1ee9ng.\n    T\u00f3m t\u1eaft ng\u1eafn g\u1ecdn: \u0110\u1ea3m b\u1ea3o ng\u01b0\u1eddi \u0111\u1ecdc d\u1ec5 d\u00e0ng hi\u1ec3u v\u00e0 \u00e1p d\u1ee5ng.\n\n    L\u01b0u \u00fd: M\u1ee5c ti\u00eau c\u1ee7a k\u1ebf ho\u1ea1ch th\u00ed \u0111i\u1ec3m l\u00e0 tri\u1ec3n khai th\u1eed nghi\u1ec7m, c\u00f3 ki\u1ec3m so\u00e1t; gi\u00fap \u0111\u00e1nh gi\u00e1 t\u00ednh hi\u1ec7u qu\u1ea3 tr\u01b0\u1edbc khi nh\u00e2n r\u1ed9ng.",
    allow_delegation=True,
    verbose=True,
    tools=[],
    llm=create_llm("OpenAI: gpt-4o-mini", 0.1)
)
            ,
        
Agent(
    role="Evaluation Agent",
    backstory="B\u1ea1n c\u00f3 m\u1ed9t b\u00e1o c\u00e1o c\u00f4ng t\u00e1c c\u00f4ng \u0111o\u00e0n n\u0103m 2023-2024, trong \u0111\u00f3 n\u00eau r\u00f5:\n\n    Nh\u1eefng k\u1ebft qu\u1ea3 t\u00edch c\u1ef1c \u0111\u00e3 \u0111\u1ea1t \u0111\u01b0\u1ee3c.\n    Nh\u1eefng h\u1ea1n ch\u1ebf, kh\u00f3 kh\u0103n, th\u00e1ch th\u1ee9c c\u00f2n t\u1ed3n t\u1ea1i.",
    goal="NHI\u1ec6M V\u1ee4 (Task)\n\n    X\u00e1c \u0111\u1ecbnh m\u1eb7t l\u1ee3i (\u01b0u \u0111i\u1ec3m) v\u00e0 m\u1eb7t h\u1ea1i (nh\u01b0\u1ee3c \u0111i\u1ec3m) ch\u00ednh t\u1eeb b\u00e1o c\u00e1o c\u00f4ng \u0111o\u00e0n.\n    D\u1ef1a tr\u00ean nh\u1eefng g\u00ec \u0111\u00e3 \u0111\u1ea1t \u0111\u01b0\u1ee3c v\u00e0 nh\u1eefng g\u00ec c\u00f2n h\u1ea1n ch\u1ebf, \u0111\u01b0a ra \u0111\u1ec1 xu\u1ea5t c\u1ee5 th\u1ec3 nh\u1eb1m c\u1ea3i thi\u1ec7n ho\u1ea1t \u0111\u1ed9ng c\u00f4ng \u0111o\u00e0n trong n\u0103m h\u1ecdc s\u1eafp t\u1edbi.\n\nH\u01af\u1edaNG D\u1eaaN AI (Instruction Steps)\n\n    T\u00f3m l\u01b0\u1ee3c th\u00f4ng tin c\u1ed1t l\u00f5i: Li\u1ec7t k\u00ea \u0111i\u1ec3m m\u1ea1nh, \u0111i\u1ec3m y\u1ebfu, c\u01a1 h\u1ed9i v\u00e0 th\u00e1ch th\u1ee9c n\u1ed5i b\u1eadt.\n    Ph\u00e2n t\u00edch nguy\u00ean nh\u00e2n: X\u00e1c \u0111\u1ecbnh l\u00fd do d\u1eabn \u0111\u1ebfn c\u00e1c k\u1ebft qu\u1ea3 t\u00edch c\u1ef1c/ti\u00eau c\u1ef1c.\n    \u0110\u01b0a ra \u0111\u1ec1 xu\u1ea5t c\u1ea3i thi\u1ec7n:\n        \u01afu ti\u00ean c\u00e1c gi\u1ea3i ph\u00e1p th\u1ef1c t\u1ebf, kh\u1ea3 thi (\u0111\u00e0o t\u1ea1o gi\u00e1o vi\u00ean, h\u1ed7 tr\u1ee3 t\u00e0i ch\u00ednh, k\u1ebft n\u1ed1i\u2026).\n        L\u00e0m r\u00f5 m\u1ee5c ti\u00eau, hi\u1ec7u qu\u1ea3 mong \u0111\u1ee3i, v\u00e0 c\u00e1ch th\u1ef1c hi\u1ec7n.\n    T\u1ed5ng h\u1ee3p ng\u1eafn g\u1ecdn: S\u1eafp x\u1ebfp \u0111\u1ec1 xu\u1ea5t th\u00e0nh danh s\u00e1ch r\u00f5 r\u00e0ng, v\u1edbi nh\u1eefng b\u01b0\u1edbc c\u1ee5 th\u1ec3 \u0111\u1ec3 tri\u1ec3n khai.",
    allow_delegation=True,
    verbose=True,
    tools=[],
    llm=create_llm("OpenAI: gpt-4o-mini", 0.1)
)
            
    ]
    return agents

def load_tasks(agents):
    tasks = [
        
Task(
    description="B\u1ed0I C\u1ea2NH (Context)\nB\u1ea1n l\u00e0 hi\u1ec7u tr\u01b0\u1edfng m\u1ed9t tr\u01b0\u1eddng THPT, c\u00f3 trong tay:\n\n    V\u0103n b\u1ea3n hi\u1ec7n c\u00f3 (b\u00e1o c\u00e1o c\u00f4ng \u0111o\u00e0n, k\u1ebf ho\u1ea1ch c\u1ee5m tr\u01b0\u1eddng, ch\u1ec9 \u0111\u1ea1o B\u1ed9/S\u1edf...)\n    K\u1ebft qu\u1ea3 ho\u1ea1t \u0111\u1ed9ng c\u1ee7a tr\u01b0\u1eddng trong n\u0103m tr\u01b0\u1edbc\n    C\u00e1c \u00fd t\u01b0\u1edfng \u0111\u00e3 brainstorm \u1edf m\u1ee9c \u0111\u1ed9 cao (high-level)\n\nNHI\u1ec6M V\u1ee4 (Task)\nH\u00e3y th\u1ef1c hi\u1ec7n c\u00e1c b\u01b0\u1edbc, theo th\u1ee9 t\u1ef1:\n\n    T\u00f3m t\u1eaft v\u0103n b\u1ea3n hi\u1ec7n c\u00f3 (b\u00e1o c\u00e1o c\u00f4ng \u0111o\u00e0n, k\u1ebf ho\u1ea1ch c\u1ee5m tr\u01b0\u1eddng\u2026).\n    Brainstorm high-level \u00fd t\u01b0\u1edfng m\u1edbi t\u1eeb K\u1ebf ho\u1ea1ch c\u1ee7a B\u1ed9/S\u1edf, k\u1ebft h\u1ee3p nh\u1eefng v\u00ed d\u1ee5 t\u1eeb tr\u01b0\u1eddng kh\u00e1c (trong/ngo\u00e0i n\u01b0\u1edbc), g\u1ee3i \u00fd \u00e1p d\u1ee5ng cho tr\u01b0\u1eddng.\n    D\u1ef1a tr\u00ean k\u1ebft qu\u1ea3 ho\u1ea1t \u0111\u1ed9ng tr\u01b0\u1edbc \u0111\u00e2y (v\u00e0 n\u1ed9i dung \u0111\u00e3 t\u00f3m t\u1eaft), x\u00e1c \u0111\u1ecbnh m\u1eb7t l\u1ee3i, m\u1eb7t h\u1ea1i \u0111\u1ec3 \u0111\u1ec1 xu\u1ea5t c\u00e1c \u0111i\u1ec3m c\u1ea7n c\u1ea3i thi\u1ec7n.\n    X\u00e2y d\u1ef1ng K\u1ebf ho\u1ea1ch th\u00ed \u0111i\u1ec3m d\u1ef1a tr\u00ean:\n        Nh\u1eefng \u0111\u1ec1 xu\u1ea5t c\u1ea3i thi\u1ec7n v\u1eeba \u0111\u01b0\u1ee3c n\u00eau.\n        Nh\u1eefng \u00fd t\u01b0\u1edfng \u0111\u00e3 brainstorm \u1edf b\u01b0\u1edbc 2.\n        Ch\u00fa tr\u1ecdng t\u00ednh kh\u1ea3 thi, c\u00f3 l\u1ed9 tr\u00ecnh, m\u1ee5c ti\u00eau, ti\u00eau ch\u00ed \u0111\u00e1nh gi\u00e1 s\u01a1 ",
    expected_output="\n    T\u00f3m l\u01b0\u1ee3c th\u00f4ng tin ch\u00ednh t\u1eeb c\u00e1c t\u00e0i li\u1ec7u.\n    Brainstorm: Li\u1ec7t k\u00ea \u00fd t\u01b0\u1edfng l\u1edbn, s\u00e1ng t\u1ea1o, c\u00f3 v\u00ed d\u1ee5 tham chi\u1ebfu.\n    Ph\u00e2n t\u00edch l\u1ee3i \u2013 h\u1ea1i, li\u1ec7t k\u00ea nh\u1eefng \u0111i\u1ec3m m\u1ea1nh/y\u1ebfu, kh\u00f3 kh\u0103n/thu\u1eadn l\u1ee3i.\n    \u0110\u1ec1 xu\u1ea5t c\u1ea3i thi\u1ec7n: C\u00e1c gi\u1ea3i ph\u00e1p gi\u00fap kh\u1eafc ph\u1ee5c nh\u01b0\u1ee3c \u0111i\u1ec3m v\u00e0 th\u00fac \u0111\u1ea9y \u01b0u \u0111i\u1ec3m.\n    K\u1ebf ho\u1ea1ch th\u00ed \u0111i\u1ec3m:\n        Ph\u1ea1m vi \u00e1p d\u1ee5ng, th\u1eddi gian, ngu\u1ed3n l\u1ef1c, c\u00e1ch tri\u1ec3n khai c\u1ee5 th\u1ec3.\n        Nh\u1eadn di\u1ec7n r\u1ee7i ro, c\u00e1ch qu\u1ea3n l\u00fd r\u1ee7i ro.\n        N\u00eau r\u00f5 ti\u00eau ch\u00ed \u0111\u00e1nh gi\u00e1 s\u01a1 b\u1ed9.\n\n    L\u01b0u \u00fd: Tu\u00e2n theo th\u1ee9 t\u1ef1 tr\u00ean \u0111\u1ec3 \u0111\u1ea3m b\u1ea3o m\u1ea1ch logic v\u00e0 \u0111\u1ed9 ch\u00ednh x\u00e1c trong qu\u00e1 tr\u00ecnh ph\u00e2n t\u00edch.",
    agent=next(agent for agent in agents if agent.role == "Pilot Planning Agent"),
    async_execution=False
)
            
    ]
    return tasks

def main():
    st.title("Planner")

    agents = load_agents()
    tasks = load_tasks(agents)
    crew = Crew(
        agents=agents, 
        tasks=tasks, 
        process="hierarchical", 
        verbose=False, 
        memory=True, 
        cache=True, 
        max_rpm=1000,
        manager_agent=next(agent for agent in agents if agent.role == "Evaluation Agent")
    )

    

    placeholders = {
        
    }
    with st.spinner("Running crew..."):
        try:
            result = crew.kickoff(inputs=placeholders)
            with st.expander("Final output", expanded=True):
                if hasattr(result, 'raw'):
                    st.write(result.raw)                
            with st.expander("Full output", expanded=False):
                st.write(result)
        except Exception as e:
            st.error(f"An error occurred: {str(e)}")

if __name__ == '__main__':
    main()
