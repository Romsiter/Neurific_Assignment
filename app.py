
import streamlit as st
import io
from io import BytesIO
from langchain.document_loaders import UnstructuredURLLoader, PyPDFLoader, TextLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter, CharacterTextSplitter
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.chains import RetrievalQA
from langchain.prompts import PromptTemplate
from langchain.embeddings import HuggingFaceEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.chains import create_history_aware_retriever, create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from PyPDF2 import PdfReader
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import openai
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.runnables.history import RunnableWithMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
import pickle
import os
import numpy as np
from dotenv import load_dotenv
from bs4 import BeautifulSoup
import requests, re, os
from langchain.schema import Document

if "last_sources" not in st.session_state:
    st.session_state.last_sources = None

if "vector_store" not in st.session_state:
    st.session_state.vector_store = None 
    
if "chat_history" not in st.session_state:
    st.session_state.chat_history = [] 
    
if "store" not in st.session_state:
    st.session_state.store = None    
       
if "memory" not in st.session_state:
    st.session_state.memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)    


# ─── 1) Load API key ───────────────────────────────────────────────────────────────
load_dotenv()
openai.api_key=os.getenv('api_key')

# ─── 2) Prepare LLM ────────────────────────────────────────────────────────────────
llm=ChatOpenAI(temperature=0, model="gpt-4",openai_api_key = openai.api_key)

def load_olah_sections(url: str):
    """Fetch Chris Olah’s blog and split by each <h2> heading."""
    resp = requests.get(url)
    soup = BeautifulSoup(resp.text, "html.parser")
    docs = []
    for h2 in soup.find_all("h2"):
        title = h2.get_text().strip()
        # gather everything until next <h2>
        texts = []
        for sib in h2.next_siblings:
            if getattr(sib, "name", None) == "h2":
                break
            texts.append(sib.get_text().strip())
        content = "\n\n".join(t for t in texts if t)
        docs.append(Document(
            page_content=content,
            metadata={
                "source": "Understanding LSTMs by Chris Olah",
                "section": title
            }
        ))
    return docs

def load_cmu_sections(pdf_url: str, local_path="LSTM.pdf"):
    """
    Download CMU PDF if needed, then split by numbered section headings like '2.1 LSTM Architecture'.
    """
    # download once
    if not os.path.exists(local_path):
        r = requests.get(pdf_url)
        with open(local_path, "wb") as f:
            f.write(r.content)

    raw = PyPDFLoader(local_path).load()
    docs = []
    heading_pattern = re.compile(r'^(?P<num>\d+(\.\d+)*)(?:\s+)(?P<title>[A-Z][^\n]+)', re.MULTILINE)
    for page in raw:
        text = page.page_content
        # find all headings and their spans
        matches = list(heading_pattern.finditer(text))
        for i, m in enumerate(matches):
            start = m.end()
            end = matches[i+1].start() if i+1 < len(matches) else len(text)
            section_name = f"{m.group('num')} {m.group('title').strip()}"
            chunk_text = text[start:end].strip()
            if chunk_text:
                docs.append(Document(
                    page_content=chunk_text,
                    metadata={
                        "source": "LSTM Notes (CMU Deep Learning, Spring 2023)",
                        "section": section_name,
                        "page": page.metadata["page"]
                    }
                ))
    return docs

# ─── 3) Build / Cache the FAISS index ─────────────────────────────────────────────
def get_text_with_metadata():
    blog_docs = load_olah_sections("https://colah.github.io/posts/2015-08-Understanding-LSTMs/")
    cmu_docs  = load_cmu_sections("https://deeplearning.cs.cmu.edu/S23/document/readings/LSTM.pdf")

    # chunk them a bit so each "section" doesn’t get too huge
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=150)
    all_docs = splitter.split_documents(blog_docs + cmu_docs)

    return all_docs

def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in store:
        store[session_id] = ChatMessageHistory()
    return store[session_id]

def clear_chat_history():
    st.session_state.memory.clear()
    st.session_state.last_sources=None
    st.session_state.chat_history=[]
    st.session_state.messages = [
        {"role": "assistant", "content": "upload some pdfs and ask me a question"}]
    st.success("Chat history cleared!")
# 3e) Embed + FAISS
embedding_model_id = "BAAI/bge-small-en-v1.5"
embeddings = HuggingFaceEmbeddings(
model_name=embedding_model_id,
        )   
#Streamlit UI 
st.title("PDF Chatbot :robot_face:")
st.subheader("Welcome to the chat!")

st.markdown(
    """
    Ask me anything about Long Short-Term Memory networks, and I'll answer
    using Chris Olah’s blog post **and** the CMU PDF—citing both source **and** section/page.
    """
)
 
vector_store = FAISS.from_documents(get_text_with_metadata(), embeddings)
st.session_state.vector_store=vector_store

if(vector_store is not None):
                retriever=vector_store.as_retriever()
                #memory = st.session_state.memory    
                ### Contextualize question ###
                contextualize_q_system_prompt = """Given a chat history and the latest user question \
                which might reference context in the chat history, formulate a standalone question \
                which can be understood without the chat history. Do NOT answer the question, \
                just reformulate it if needed and otherwise return it as is."""
                contextualize_q_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", contextualize_q_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                history_aware_retriever = create_history_aware_retriever(
                    llm, retriever, contextualize_q_prompt
                )
                qa_system_prompt = """You are an assistant for question-answering tasks. \
                Use the following pieces of retrieved context to answer the question. \
                If you don't find the information to answer the question present in the retrieved context, just output 'Sorry, I didn’t understand your question. Do you want to connect with a live agent?' and nothing else. \
                    
                {context}"""

                qa_prompt = ChatPromptTemplate.from_messages(
                    [
                        ("system", qa_system_prompt),
                        MessagesPlaceholder("chat_history"),
                        ("human", "{input}"),
                    ]
                )
                question_answer_chain = create_stuff_documents_chain(llm, qa_prompt)

                rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

                store = {}

                conversational_rag_chain = RunnableWithMessageHistory(
                    rag_chain,
                    get_session_history,
                    input_messages_key="input",
                    history_messages_key="chat_history",
                    output_messages_key="answer",
                )
                st.session_state.conversational_rag_chain = conversational_rag_chain
                st.session_state.store=store

query = st.text_input("❓ Your question:")
if st.button("Ask"):
    if not query:
        st.warning("Please enter a question!")
    else:
        conversational_rag_chain = st.session_state.conversational_rag_chain
        with st.spinner("🔄 Retrieving & generating…"):
            result = conversational_rag_chain.invoke(
                        {"input": query},
                        config={
                            "configurable": {"session_id": "abc123"}
                        },
                    )
            answer = result['answer']
            sources= result['context']
            st.markdown("### 💡 Answer")
            st.write(answer)
            for i, doc in enumerate(sources, start=1):
                src    = doc.metadata.get("source", "Unknown source")
                page   = doc.metadata.get("page")
                sect   = doc.metadata.get("section")

                # Build a nice label
                if page is not None:
                    label = f"""{src} — Page {page} \n
                                Section: {sect}
                            """
                elif sect is not None:
                    label = f"""{src}
                    
                                Section: {sect}
                             """
                else:
                    label = src

                # You can tuck the snippet into an expander if you like
                with st.expander(f"Source {i}: {label}", expanded=False):
                    st.write(doc.page_content)
