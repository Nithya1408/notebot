import streamlit as st
from streamlit import sidebar
from PyPDF2 import PdfReader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate

st.header("Narni")
with sidebar:
    st.title("My Notes")
    file = st.file_uploader("Kindly upload your notes PDF and start asking questions", type="pdf")

# extracting text from the uploaded file
if file is not None:
    my_pdf = PdfReader(file)
    text = ""
    for page in my_pdf.pages:
        text += page.extract_text() or ""

    if not text.strip():
        st.error("No text could be extracted. Is this a scanned image PDF?")
        st.stop()

    # break it into chunks
    splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=150, length_function=len)
    chunks = splitter.split_text(text)

    # free local embeddings
    embeddings = HuggingFaceEmbeddings(model_name="all-MiniLM-L6-v2")

    # creating VectorDB and storing embeddings into it
    vector_store = FAISS.from_texts(chunks, embeddings)

    # get user query
    user_query = st.text_input("Enter your query")

    # semantic search from vector store
    if user_query:
        matching_chunks = vector_store.similarity_search(user_query, k=6)

        # combine the matching chunks into one block of context
        context_text = "\n\n".join(chunk.page_content for chunk in matching_chunks)

        # free local LLM (make sure the Ollama app is running)
        llm = ChatOllama(model="llama3.2", temperature=0)

        # the prompt template — this is where you control how Narni answers
        prompt = ChatPromptTemplate.from_template(
            """You are Narni, a helpful UPSC study tutor.
Answer the question using ONLY the context from the student's notes below.
If the answer is not in the context, simply say "I couldn't find that in your notes."

Context:
{context}

Question: {question}
"""
        )

        # build the chain: prompt -> LLM
        chain = prompt | llm
        output = chain.invoke({"context": context_text, "question": user_query})
        st.write(output.content)