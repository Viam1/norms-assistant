import os
import streamlit as st
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings, HuggingFaceEndpoint
from langchain_community.vectorstores import Chroma
from langchain.chains import RetrievalQA
from huggingface_hub import login

# Заголовок приложения
st.title("📋 Помощник по нормам проектирования")
st.write("Задайте вопрос — ответ будет сформирован строго по загруженным документам.")

# Боковая панель для токена (чтобы не хардкодить)
with st.sidebar:
    st.header("Настройки")
    hf_token = st.text_input("Hugging Face Token (необязательно)", type="password", 
                              help="Получите токен на huggingface.co/settings/tokens")
    if hf_token:
        login(token=hf_token)

# Функция загрузки документов (кэшируется)
@st.cache_resource
def load_documents():
    docs_dir = "./docs"
    if not os.path.exists(docs_dir):
        st.error("Папка docs не найдена. Загрузите PDF в репозиторий.")
        return None
    
    loader = PyPDFDirectoryLoader(docs_dir)
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=100
    )
    texts = text_splitter.split_documents(documents)
    
    embeddings = HuggingFaceEmbeddings(
        model_name="intfloat/multilingual-e5-small"
    )
    
    vectorstore = Chroma.from_documents(
        documents=texts,
        embedding=embeddings,
        persist_directory="./chroma_db"
    )
    return vectorstore

# Инициализация
with st.spinner("Загрузка документов..."):
    vectorstore = load_documents()

if vectorstore:
    retriever = vectorstore.as_retriever(search_kwargs={"k": 3})
    
    # Настройка модели
    llm = HuggingFaceEndpoint(
        repo_id="Qwen/Qwen2.5-7B-Instruct",
        temperature=0.1,
        max_new_tokens=512,
        huggingfacehub_api_token=hf_token if hf_token else None
    )
    
    qa_chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True
    )
    
    # Поле ввода вопроса
    question = st.text_input("Ваш вопрос:", placeholder="Например: Какие требования к высоте потолков?")
    
    if question:
        with st.spinner("Поиск ответа..."):
            try:
                result = qa_chain.invoke({"query": question})
                answer = result["result"]
                sources = result["source_documents"]
                
                st.subheader("Ответ:")
                st.write(answer)
                
                st.subheader("Источники:")
                for i, doc in enumerate(sources):
                    source_name = os.path.basename(doc.metadata.get("source", "Неизвестный файл"))
                    page = doc.metadata.get("page", "N/A")
                    st.write(f"- {source_name} (стр. {page})")
            except Exception as e:
                st.error(f"Ошибка: {str(e)}")
else:
    st.error("Не удалось загрузить документы.")
