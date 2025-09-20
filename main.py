# main.py
from fastapi import FastAPI, UploadFile, File, Form
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import io
from pypdf import PdfReader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import FAISS
import os
import google.generativeai as genai

# --- Environment and Model Setup ---
# Load keys from the environment
HUGGINGFACE_TOKEN = os.environ.get("HUGGINGFACEHUB_API_TOKEN")
GOOGLE_API_KEY = os.environ.get("GOOGLE_API_KEY")
genai.configure(api_key=GOOGLE_API_KEY)
embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

# --- In-memory "database" ---
vector_store = None

# --- FastAPI Application ---
app = FastAPI(
    title="Smart Research Assistant API",
    description="An API for generating structured, evidence-based reports.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.get("/")
async def read_root():
    return {"message": "Welcome to the Smart Research Assistant API!"}

@app.post("/uploadfile/")
async def create_upload_file(file: UploadFile = File(...)):
    global vector_store
    if file.content_type != "application/pdf":
        return JSONResponse(status_code=400, content={"message": "Please upload a PDF file."})
    try:
        file_content = await file.read()
        pdf_reader = PdfReader(io.BytesIO(file_content))
        text = "".join(page.extract_text() or "" for page in pdf_reader.pages)
        if not text:
            return JSONResponse(status_code=400, content={"message": "Could not extract text from the PDF."})
        
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        chunks = text_splitter.split_text(text=text)
        vector_store = FAISS.from_texts(chunks, embedding=embeddings)
        return {"message": f"File '{file.filename}' processed successfully. You can now ask questions."}
    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})

@app.post("/ask/")
async def ask_question(question: str = Form(...)):
    global vector_store
    if not vector_store:
        return JSONResponse(status_code=400, content={"message": "No file has been uploaded yet."})
    
    if not question:
        return JSONResponse(status_code=400, content={"message": "Please provide a question."})

    try:
        docs = vector_store.similarity_search(query=question, k=3)
        context = "\n".join([doc.page_content for doc in docs])
        
        prompt_template = f"""
        You are a Smart Research Assistant. Your task is to generate a structured, evidence-based report in response to a user's question.
        Use ONLY the following context provided to answer the question. Do not use any external knowledge.
        
        CONTEXT:
        {context}
        
        QUESTION:
        {question}
        
        INSTRUCTIONS:
        1.  Analyze the context and extract the key insights that directly answer the question.
        2.  Present the answer in a clear, concise report format.
        3.  Start with a section titled "Key Takeaways" summarizing the main points.
        4.  Follow with a section titled "Sources" that includes direct quotes from the context that support your answer. Cite them clearly.
        5.  If the context does not contain the answer, state that "The provided document does not contain information on this topic."
        """
        
        model = genai.GenerativeModel('gemini-1.5-flash-latest')
        response = model.generate_content(prompt_template)
        
        return {"report": response.text}

    except Exception as e:
        return JSONResponse(status_code=500, content={"message": f"An error occurred: {str(e)}"})