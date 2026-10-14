import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
from chromadb.config import Settings
import requests
import json

# Заголовок
st.title("📋 Помощник по нормам проектирования")
st.write("Задайте вопрос — ответ будет найден в загруженных документах.")

# Функция для извлечения текста из PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        text += page.extract_text() + "\n"
    return text

# Функция для разбиения текста на фрагменты
def split_text(text, chunk_size=1000, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Загрузка и индексация документов
@st.cache_resource
def load_and_index_documents():
    docs_dir = "./docs"
    
    if not os.path.exists(docs_dir):
        return None, None
    
    # Инициализация ChromaDB
    client = chromadb.Client(Settings(
        chroma_db_impl="duckdb+parquet",
        persist_directory="./chroma_db"
    ))
    
    # Создаем или получаем коллекцию
    collection = client.get_or_create_collection(name="norms_documents")
    
    # Если коллекция уже заполнена, возвращаем её
    if collection.count() > 0:
        return client, collection
    
    # Загружаем модель эмбеддингов
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    
    # Обрабатываем все PDF
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    for filename in os.listdir(docs_dir):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(docs_dir, filename)
            text = extract_text_from_pdf(pdf_path)
            chunks = split_text(text)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                all_metadatas.append({
                    "source": filename,
                    "chunk_id": i
                })
                all_ids.append(f"{filename}_{i}")
    
    # Создаем эмбеддинги
    embeddings = model.encode(all_chunks)
    
    # Добавляем в ChromaDB
    collection.add(
        documents=all_chunks,
        embeddings=embeddings.tolist(),
        metadatas=all_metadatas,
        ids=all_ids
    )
    
    # Сохраняем
    client.persist()
    
    return client, collection

# Функция для поиска релевантных фрагментов
def search_documents(collection, query, model, n_results=3):
    query_embedding = model.encode([query]).tolist()
    
    results = collection.query(
        query_embeddings=query_embedding,
        n_results=n_results
    )
    
    return results

# Функция для получения ответа от модели
def get_answer_from_model(context, question):
    # Используем бесплатную модель через HuggingFace Inference API
    API_URL = "https://api-inference.huggingface.co/models/Qwen/Qwen2.5-7B-Instruct"
    
    prompt = f"""Используй только следующую информацию из документов для ответа на вопрос. Если в документах нет ответа, скажи об этом.

Контекст из документов:
{context}

Вопрос: {question}

Ответ:"""
    
    payload = {
        "inputs": prompt,
        "parameters": {
            "max_new_tokens": 500,
            "temperature": 0.1,
            "return_full_text": False
        }
    }
    
    headers = {"Content-Type": "application/json"}
    
    try:
        response = requests.post(API_URL, headers=headers, json=payload, timeout=60)
        result = response.json()
        
        if isinstance(result, list) and len(result) > 0:
            return result[0].get('generated_text', 'Не удалось получить ответ')
        else:
            return "Модель временно недоступна. Попробуйте через минуту."
    except Exception as e:
        return f"Ошибка при запросе к модели: {str(e)}"

# Основная логика
with st.spinner("Загрузка и индексация документов..."):
    client, collection = load_and_index_documents()

if client and collection:
    st.success(f"✅ Загружено документов: {len(os.listdir('./docs'))}")
    
    # Загружаем модель для эмбеддингов
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    
    # Поле ввода
    question = st.text_input("Ваш вопрос:", placeholder="Например: Какие требования к высоте потолков?")
    
    if question:
        with st.spinner("Поиск ответа в документах..."):
            # Ищем релевантные фрагменты
            results = search_documents(collection, question, model, n_results=3)
            
            if results['documents'] and len(results['documents'][0]) > 0:
                # Формируем контекст
                context = "\n\n".join(results['documents'][0])
                
                # Получаем ответ от модели
                answer = get_answer_from_model(context, question)
                
                # Показываем ответ
                st.subheader("📝 Ответ:")
                st.write(answer)
                
                # Показываем источники
                st.subheader("📚 Источники:")
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    source = metadata.get('source', 'Неизвестный файл')
                    st.write(f"**{i+1}. {source}**")
                    with st.expander("Показать фрагмент"):
                        st.text(doc[:500] + "..." if len(doc) > 500 else doc)
            else:
                st.warning("Не удалось найти релевантную информацию в документах.")
else:
    st.error("❌ Папка docs не найдена или пуста. Загрузите PDF-файлы в репозиторий.")
