import os
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from dotenv import load_dotenv

load_dotenv()
key = os.getenv("ACTIVE_SCHOLAR_GOOGLE_API_KEY")

# Test Embedding
try:
    print("Testing models/text-embedding-004...")
    emb = GoogleGenerativeAIEmbeddings(model="models/text-embedding-004", google_api_key=key)
    vec = emb.embed_query("Foo")
    print("Success: models/text-embedding-004 (dim", len(vec), ")")
except Exception as e:
    print("Failed models/text-embedding-004:", e)

    try:
        print("Testing models/gemini-embedding-001...")
        emb = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=key)
        vec = emb.embed_query("Foo")
        print("Success: models/gemini-embedding-001 (dim", len(vec), ")")
    except Exception as e2:
        print("Failed models/gemini-embedding-001:", e2)

# Test Chat
try:
    llm = ChatGoogleGenerativeAI(model="models/gemini-2.0-flash", google_api_key=key)
    print("Testing models/gemini-2.0-flash...")
    res = llm.invoke("Hi")
    print("Success:", res.content)
except Exception as e:
    print("Failed models/gemini-2.0-flash:", e)
