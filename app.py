import streamlit as st
import numpy as np
import faiss
import pickle
import requests

@st.cache_resource
def load_rag():
    try:
        index = faiss.read_index("rag_index.faiss")
        doc_names = np.load("doc_names.npy", allow_pickle=True).tolist()
        chunks_list = np.load("chunks_list.npy", allow_pickle=True).tolist()
        st.success(f"Loaded: {len(doc_names)} docs, {len(chunks_list)} chunks")
        return index, doc_names, chunks_list
    except Exception as e:
        st.error(f"Load error: {e}")
        return None, [], []

index, doc_names, chunks_list = load_rag()

if index is None:
    st.stop()

# Your get_embedding function
def get_embedding(text, model="embeddinggemma"):
    url = 'http://localhost:11434/api/embeddings'
    payload = {
        "model": model,
        "prompt": text
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return np.array(response.json()['embedding'], dtype=np.float32)
    else:
        raise ValueError(f"Embedding failed: {response.text}")

# Your retrieve_chunks function
def retrieve_chunks(query, index, chunks_list, doc_names, doc_filter=None, k=3):
    query_emb = get_embedding(query)
    distances, indices = index.search(query_emb.reshape(1, -1), k * 4)  # Oversample
    relevant_chunks = []
    for idx in indices[0]:
        if idx == -1:
            continue
        chunk = chunks_list[idx]
        doc_name = doc_names[idx]
        if doc_filter is None or doc_filter in doc_name:
            relevant_chunks.append(chunk)
        if len(relevant_chunks) >= k:
            break
    return relevant_chunks[:k]

# Your generate_response function
def generate_response(query, context, model="gpt-oss:120b"):  # Your 120b model
    url = 'http://localhost:11434/api/chat'
    prompt = f"""You are a financial analyst. Answer ONLY from the context. Be precise with numbers.

CRITICAL: Parse tables/JSON explicitly:
- Scan for rows like 'Net Profit', 'Net Income', 'EBITDA'.
- Handle splits: If 'Net Profit\n14,813\n16,542\n22,216'.
- Use the table headers from the documents in your answer.
- Double-check markdown columns (align by position).
- If truly absent, say "Not available" with evidence.

Context (raw text/tables/JSON):
{context}

Query: {query}

Answer (cite document name and table/page if possible):"""
    payload = {
        "model": model,
        "temperature": 0.1,  # Lower for factual extraction
        "stream": False,
        "messages": [
            {"role": "system", "content": "Stick to context; extract numbers accurately from tables."},
            {"role": "user", "content": prompt}
        ]
    }
    response = requests.post(url, json=payload)
    if response.status_code == 200:
        return response.json()['message']['content']
    else:
        raise ValueError(f"Generation failed: {response.text}")

st.title("RAG Query App")

# Sidebar
st.sidebar.title("Filters")
unique_docs = ['All Documents'] + sorted(list(set(doc_names)))
doc_filter = st.sidebar.selectbox("Select Document:", unique_docs)

# Query
query = st.text_input("Enter your query:", placeholder="e.g., BIM investment reasons")

if st.button("Run Query"):
    if query:
        with st.spinner("Retrieving..."):
            retrieved = retrieve_chunks(query, index, chunks_list, doc_names, doc_filter=doc_filter if doc_filter != 'All Documents' else None, k=3)
            context = "\n\n".join(retrieved)
            response = generate_response(query, context)
            st.subheader("Response:")
            st.write(response)
            st.subheader("Context Preview:")
            st.write(context[:800] + "..." if len(context) > 800 else context)
    else:
        st.warning("Enter a query.")

st.caption("Powered by Ollama & FAISS")
