from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from typing import Optional
from uuid import uuid4
import os
import fitz  # PyMuPDF
import requests
from tempfile import NamedTemporaryFile
import cohere
from dotenv import load_dotenv
from sklearn.metrics.pairwise import cosine_similarity

load_dotenv()
app = FastAPI()

# Cohere client setup
COHERE_API_KEY = os.getenv("COHERE_API_KEY")
if not COHERE_API_KEY:
    raise ValueError("Missing COHERE_API_KEY in environment")
co = cohere.Client(COHERE_API_KEY)

# In-memory session store
session_store = {}

# ---------- Helper Functions ----------

def download_pdf(url):
    try:
        headers = {"User-Agent": "Mozilla/5.0"}  # handle S3 / secured PDFs
        response = requests.get(url, headers=headers)
        response.raise_for_status()
        with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
            tmp.write(response.content)
            return tmp.name
    except Exception as e:
        raise ValueError(f"Failed to download PDF: {e}")

def pdf_to_text(path):
    try:
        doc = fitz.open(path)
        return "\n".join(page.get_text() for page in doc)
    except Exception as e:
        raise ValueError(f"Failed to read PDF: {e}")

def chunk_text(text, max_tokens=300):
    words = text.split()
    return [" ".join(words[i:i + max_tokens]) for i in range(0, len(words), max_tokens)]

def embed_chunks(chunks):
    response = co.embed(texts=chunks, model="embed-english-v3.0", input_type="search_document")
    return response.embeddings

def get_top_chunks(question, chunks, embeddings, top_k=5):
    query_embedding = co.embed(
        texts=[question], model="embed-english-v3.0", input_type="search_query"
    ).embeddings[0]
    sims = cosine_similarity([query_embedding], embeddings)[0]
    top_indices = sims.argsort()[::-1][:top_k]
    return [chunks[i] for i in top_indices]

def generate_answer(context, question):
    prompt = f"""You are a helpful assistant. Use the context below to answer the question strictly based on that context.

Context:
{context}

Question: {question}
Answer:"""
    response = co.generate(
        model="command-r-plus",
        prompt=prompt,
        max_tokens=500,
        temperature=0.3,
        stop_sequences=["--"]
    )
    return response.generations[0].text.strip()

# ---------- API Endpoints ----------

@app.post("/load")
async def load_pdf(
    pdf_url: Optional[str] = Form(None),
    file: Optional[UploadFile] = File(None),
):
    try:
        if pdf_url:
            pdf_path = download_pdf(pdf_url)
        elif file:
            with NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
                tmp.write(await file.read())
                pdf_path = tmp.name
        else:
            return JSONResponse({"error": "Provide either a PDF URL or a file."}, status_code=400)

        text = pdf_to_text(pdf_path)
        chunks = chunk_text(text)
        embeddings = embed_chunks(chunks)

        session_id = str(uuid4())
        session_store[session_id] = {
            "chunks": chunks,
            "embeddings": embeddings
        }

        return JSONResponse({"message": "PDF processed successfully.", "session_id": session_id})

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

@app.post("/chat")
async def chat_with_pdf(
    session_id: str = Form(...),
    question: str = Form(...)
):
    try:
        session = session_store.get(session_id)
        if not session:
            return JSONResponse({"error": "Invalid session_id."}, status_code=400)

        chunks = session["chunks"]
        embeddings = session["embeddings"]
        top_chunks = get_top_chunks(question, chunks, embeddings)
        context = "\n\n".join(top_chunks)
        answer = generate_answer(context, question)

        return JSONResponse({
            "session_id": session_id,
            "question": question,
            "answer": answer
        })

    except Exception as e:
        return JSONResponse({"error": str(e)}, status_code=500)

# ---------- Optional: Uvicorn Run ----------

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("main:app", host="0.0.0.0", port=8000)
