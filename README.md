# 🧠 Retrieval-Augmented Generation (RAG) System using LangChain & LangGraph

This project demonstrates how to build a complete **Retrieval-Augmented Generation (RAG)** pipeline using **LangChain**, **LangGraph**, and embeddings from **OpenAI**. The system loads a custom knowledge base, splits it into chunks, stores it in a vector database, retrieves relevant documents for a query, and generates a grounded and accurate answer using an LLM.

---

## 🚀 Features

✅ Fully functional RAG pipeline
✅ Uses **LangGraph** for modular, node-based workflows
✅ Embedding-based retrieval using **OpenAI Embeddings**
✅ Vector store via **InMemoryVectorStore** (can switch to Chroma/Pinecone/Qdrant easily)
✅ Clean separation of nodes: **Retrieve → Generate → Output**
✅ JSON-based knowledge base (`knowledge_base.json`)
✅ Extensible architecture for multi-hop retrieval, reranking, or evaluation

---

## 📁 Project Structure

```
│── knowledge_base.json      # Your custom knowledge data
│── rag_system.ipynb         # Main notebook / script
│── README.md                # Project documentation
```

---

## 🛠️ Installation

Install all required packages:

```bash
pip install -U langchain langchain-core langchain-community langchain-openai langchain-text-splitters langgraph chromadb
```

> If you want Pinecone/Qdrant instead:

```
pip install -U langchain-pinecone pinecone-client
pip install -U qdrant-client
```

---

## 🔑 Set your API Key

```python
import os
os.environ["OPENAI_API_KEY"] = "sk-your-key-here"
```

---

## 📚 Load and Prepare Your Knowledge Base

```python
import json
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter

with open("knowledge_base.json") as f:
    items = json.load(f)

docs = [
    Document(page_content=item["text"], metadata={"id": item.get("id", i)})
    for i, item in enumerate(items)
]

splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=160)
chunks = splitter.split_documents(docs)
```

---

## 🔍 Create Vector Store

```python
from langchain_openai import OpenAIEmbeddings
from langchain_core.vectorstores import InMemoryVectorStore

embeddings = OpenAIEmbeddings(model="text-embedding-3-large")
vector_store = InMemoryVectorStore(embeddings)

vector_store.add_documents(chunks)
retriever = vector_store.as_retriever(search_kwargs={"k": 5})
```

---

## 🧩 Build RAG Workflow with LangGraph

```python
from typing_extensions import TypedDict, List
from langchain_core.documents import Document
from langgraph.graph import StateGraph, END
from langchain_openai import ChatOpenAI

class State(TypedDict):
    question: str
    context: List[Document]
    answer: str

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)

def retrieve(state: State):
    docs = retriever.get_relevant_documents(state["question"])
    return {"context": docs}

def generate(state: State):
    ctx = "\n\n".join([d.page_content for d in state["context"]])
    q = state["question"]

    prompt = f"Use ONLY the following context to answer:\n\n{ctx}\n\nQuestion: {q}\nAnswer:"
    answer = llm.invoke(prompt).content
    return {"answer": answer}

graph = StateGraph(State)
graph.add_node("retrieve", retrieve)
graph.add_node("generate", generate)
graph.set_entry_point("retrieve")
graph.add_edge("retrieve", "generate")
graph.add_edge("generate", END)

app = graph.compile()
```

---

## 💬 Run the RAG System

```python
response = app.invoke({"question": "What does the knowledge base say about AI?"})
print(response["answer"])
```

---

## ✅ Example `knowledge_base.json`

```json
[
  {
    "id": "rag-basic",
    "source": "rag",
    "text": "Retrieval-Augmented Generation combines document retrieval with LLM generation to produce accurate, grounded responses."
  },
  {
    "id": "chunking",
    "source": "rag",
    "text": "Chunking breaks large text into manageable segments to improve embedding and retrieval performance."
  }
]
```

---

## 🧱 Customize or Extend

You can easily add:

✅ **Reranking Node**
✅ **Document Grading Node**
✅ **Chat History Memory Node**
✅ **Multi-hop Retrieval**
✅ **Gradio or Streamlit UI**
✅ **Switch between Chroma, Pinecone, Qdrant**

Just modify the graph with more nodes:

```
retrieve → grade → rerank → generate → answer
```

---

## 🎯 Conclusion

This RAG system demonstrates a clean, modular approach to retrieval-augmented generation using LangChain and LangGraph. By building on a node-based workflow and vector search, it produces grounded and reliable answers while remaining extensible for advanced AI applications.
