import os
from typing import List, Dict, Any, Generator
from dotenv import load_dotenv

load_dotenv()

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")


def _build_prompt(query: str, chunks: List[Dict[str, Any]]) -> str:
    """Build the prompt from query and chunks."""
    if not chunks:
        return "No relevant information found in the document."
    
    context = "\n\n".join([f"[Chunk {c['chunk_id']}]: {c['text']}" for c in chunks])
    
    return f"""Answer the question based ONLY on the provided context. If the answer is not in the context, say "I cannot find this information in the document."

Context:
{context}

Question: {query}

Answer:"""


def generate_answer(query: str, chunks: List[Dict[str, Any]], model: str = "cascade") -> str:
    prompt = _build_prompt(query, chunks)
    
    if model == "groq":
        return _call_groq(prompt)
    elif model == "gemini":
        return _call_gemini(prompt)
    elif model == "cascade":
        return _call_cascade(prompt)
    else:
        raise ValueError(f"Unknown model: {model}. Use 'groq', 'gemini', or 'cascade'.")


def generate_answer_stream(query: str, chunks: List[Dict[str, Any]], model: str = "cascade") -> Generator[str, None, None]:
    prompt = _build_prompt(query, chunks)
    
    if model == "groq":
        yield from _call_groq_stream(prompt)
    elif model == "gemini":
        yield from _call_gemini_stream(prompt)
    elif model == "cascade":
        yield from _call_cascade_stream(prompt)
    else:
        yield f"Unknown model: {model}. Use 'groq', 'gemini', or 'cascade'."


def _call_groq(prompt: str) -> str:
    """Call Groq API with llama3-70b-8192 (non-streaming)."""
    if not GROQ_API_KEY:
        return "⚠️ GROQ_API_KEY not set in .env file."
    
    try:
        import httpx
        from groq import Groq
        
        http_client = httpx.Client(timeout=60.0)
        client = Groq(api_key=GROQ_API_KEY, http_client=http_client)
        
        response = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
        )
        return response.choices[0].message.content.strip()
    except Exception as e:
        return f"⚠️ Groq error: {str(e)}"


def _call_groq_stream(prompt: str) -> Generator[str, None, None]:
    """Call Groq API with streaming."""
    if not GROQ_API_KEY:
        yield "⚠️ GROQ_API_KEY not set in .env file."
        return
    
    try:
        import httpx
        from groq import Groq
        
        http_client = httpx.Client(timeout=60.0)
        client = Groq(api_key=GROQ_API_KEY, http_client=http_client)
        
        stream = client.chat.completions.create(
            model="llama3-70b-8192",
            messages=[{"role": "user", "content": prompt}],
            temperature=0.1,
            max_tokens=1024,
            stream=True,
        )
        
        for chunk in stream:
            if chunk.choices[0].delta.content:
                yield chunk.choices[0].delta.content
    except Exception as e:
        yield f"⚠️ Groq error: {str(e)}"


def _call_gemini(prompt: str) -> str:
    """Call Gemini API with gemini-2.0-flash (non-streaming)."""
    if not GEMINI_API_KEY:
        return "⚠️ GEMINI_API_KEY not set in .env file."
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
            )
        )
        return response.text.strip()
    except Exception as e:
        return f"⚠️ Gemini error: {str(e)}"


def _call_gemini_stream(prompt: str) -> Generator[str, None, None]:
    """Call Gemini API with streaming."""
    if not GEMINI_API_KEY:
        yield "⚠️ GEMINI_API_KEY not set in .env file."
        return
    
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        
        model = genai.GenerativeModel("gemini-2.0-flash")
        response = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.1,
                max_output_tokens=1024,
            ),
            stream=True,
        )
        
        for chunk in response:
            if chunk.text:
                yield chunk.text
    except Exception as e:
        yield f"⚠️ Gemini error: {str(e)}"


def _call_cascade(prompt: str) -> str:
    """
    Cascade: Groq generates a draft → Gemini refines/structures the answer.
    Falls back to the other model if one fails.
    """
    draft = None

    if GROQ_API_KEY:
        draft = _call_groq(prompt)
        if draft.startswith("⚠️"):
            draft = None

    if GEMINI_API_KEY:
        refine_prompt = (
            f"Refine and structure the following answer to make it clear, "
            f"well-organized, and professional. Keep all factual information "
            f"intact — do not add or remove any facts.\n\n"
            f"Draft Answer:\n{draft if draft else '(no draft available)'}\n\n"
            f"Structured Answer:"
        ) if draft else prompt

        result = _call_gemini(refine_prompt)
        if not result.startswith("⚠️"):
            return result

    if draft:
        return draft

    return "⚠️ Both Groq and Gemini failed to generate an answer."


def _call_cascade_stream(prompt: str) -> Generator[str, None, None]:
    """
    Cascade streaming: Groq generates full draft silently, then Gemini
    streams the refined/structured answer token by token.
    """
    draft = None

    if GROQ_API_KEY:
        try:
            import httpx
            from groq import Groq

            http_client = httpx.Client(timeout=60.0)
            client = Groq(api_key=GROQ_API_KEY, http_client=http_client)

            response = client.chat.completions.create(
                model="llama3-70b-8192",
                messages=[{"role": "user", "content": prompt}],
                temperature=0.1,
                max_tokens=1024,
            )
            draft = response.choices[0].message.content.strip()
        except Exception:
            draft = None

    if GEMINI_API_KEY:
        refine_prompt = (
            f"Refine and structure the following answer to make it clear, "
            f"well-organized, and professional. Keep all factual information "
            f"intact — do not add or remove any facts.\n\n"
            f"Draft Answer:\n{draft if draft else '(no draft available)'}\n\n"
            f"Structured Answer:"
        ) if draft else prompt

        yield from _call_gemini_stream(refine_prompt)
        return

    if draft:
        yield draft
        return

    yield "⚠️ Both Groq and Gemini failed to generate an answer."