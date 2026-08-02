import os
from dotenv import load_dotenv
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.prompts import ChatPromptTemplate

load_dotenv()


vector_store = None


def process_and_index_document(file_path: str) -> str:
    """Loads a PDF or DOCX file, splits it into chunks, and indexes it into Chroma DB."""
    global vector_store

    
    ext = os.path.splitext(file_path)[-1].lower()
    if ext == ".pdf":
        loader = PyPDFLoader(file_path)
    elif ext in [".docx", ".doc"]:
        loader = Docx2txtLoader(file_path)
    else:
        return "Unsupported file type. Please upload a PDF or DOCX file."

    docs = loader.load()

    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    chunks = text_splitter.split_documents(docs)

    
    embeddings = OpenAIEmbeddings()
    vector_store = Chroma.from_documents(chunks, embeddings)

    return f"Successfully processed and indexed '{os.path.basename(file_path)}' ({len(chunks)} text chunks created)."


def answer_rag_question(query: str) -> str:
    """Retrieves relevant context from vector store and answers strictly based on document content."""
    global vector_store

    if vector_store is None:
        return "Please upload and process a document first before asking questions."

    
    retriever = vector_store.as_retriever(search_kwargs={"k": 3})
    relevant_docs = retriever.invoke(query)

    context_text = "\n\n---\n\n".join([doc.page_content for doc in relevant_docs])

    
    prompt = ChatPromptTemplate.from_template("""
You are a helpful assistant that answers questions strictly based on the provided document context.

Document Context:
{context}

Question:
{question}

Rules:
- Answer using ONLY the information provided in the Document Context.
- If the answer is not mentioned in the context, explicitly say: "I cannot find the answer to that question in the uploaded document."
- Keep the answer clear and concise.
""")

    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
    chain = prompt | llm

    response = chain.invoke({"context": context_text, "question": query})
    return response.content