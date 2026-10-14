import os
import streamlit as st
from pypdf import PdfReader
from sentence_transformers import SentenceTransformer
import chromadb
import requests

# Заголовок
st.title("📋 Помощник по нормам проектирования")
st.write("Задайте вопрос — ответ будет найден в загруженных документах.")

# Функция для извлечения текста из PDF
def extract_text_from_pdf(pdf_path):
    reader = PdfReader(pdf_path)
    text = ""
    for page in reader.pages:
        page_text = page.extract_text()
        if page_text:
            text += page_text + "\n"
    return text

# Функция для разбиения текста на фрагменты
def split_text(text, chunk_size=800, overlap=100):
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end]
        if chunk.strip():
            chunks.append(chunk)
        start += chunk_size - overlap
    return chunks

# Загрузка и индексация документов
@st.cache_resource
def load_and_index_documents():
    docs_dir = "./docs"
    
    if not os.path.exists(docs_dir):
        return None, None, None
    
    # Инициализация ChromaDB (новый API)
    client = chromadb.PersistentClient(path="./chroma_db")
    
    # Создаем или получаем коллекцию
    collection = client.get_or_create_collection(
        name="norms_documents",
        metadata={"hnsw:space": "cosine"}
    )
    
    # Если коллекция уже заполнена, возвращаем её
    if collection.count() > 0:
        st.info(f"База данных уже содержит {collection.count()} фрагментов.")
        return client, collection, None
    
    # Загружаем модель эмбеддингов
    model = SentenceTransformer('intfloat/multilingual-e5-small')
    
    # Обрабатываем все PDF
    all_chunks = []
    all_metadatas = []
    all_ids = []
    
    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        return None, None, None
    
    for filename in pdf_files:
        pdf_path = os.path.join(docs_dir, filename)
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            continue
            
        chunks = split_text(text)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append(chunk)
            all_metadatas.append({
                "source": filename,
                "chunk_id": i
            })
            all_ids.append(f"{filename}_{i}")
    
    if not all_chunks:
        return None, None, None
    
    # Создаем эмбеддинги
    embeddings = model.encode(all_chunks, show_progress_bar=True).tolist()
    
    # Добавляем в ChromaDB (батчами по 100)
    batch_size = 100
    for i in range(0, len(all_chunks), batch_size):
        end = i + batch_size
        collection.add(
            documents=all_chunks[i:end],
            embeddings=embeddings[i:end],
            metadatas=all_metadatas[i:end],
            ids=all_ids[i:end]
        )
    
    st.success(f"✅ Проиндексировано {len(all_chunks)} фрагментов из {len(pdf_files)} PDF.")
    
    return client, collection, model

# Функция для получения ответа от модели
def get_answer_from_model(context, question):
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
        elif isinstance(result, dict) and 'error' in result:
            return f"Модель загружается: {result['error']}. Попробуйте через минуту."
        else:
            return "Модель временно недоступна. Попробуйте через минуту."
    except Exception as e:
        return f"Ошибка при запросе к модели: {str(e)}"

# Основная логика
with st.spinner("Загрузка и индексация документов..."):
    client, collection, model = load_and_index_documents()

if client and collection:
    # Загружаем модель для эмбеддингов (если не была загружена)
    if model is None:
        model = SentenceTransformer('intfloat/multilingual-e5-small')
    
    # Поле ввода
    question = st.text_input("Ваш вопрос:", placeholder="Например: Какие требования к высоте потолков?")
    
    if question:
        with st.spinner("Поиск ответа в документах..."):
            # Создаем эмбеддинг запроса
            query_embedding = model.encode([question]).tolist()
            
            # Ищем релевантные фрагменты
            results = collection.query(
                query_embeddings=query_embedding,
                n_results=3,
                include=["documents", "metadatas"]
            )
            
            if results['documents'] and len(results['documents'][0]) > 0:
                # Формируем контекст
                context = "\n\n---\n\n".join(results['documents'][0])
                
                # Получаем ответ от модели
                answer = get_answer_from_model(context, question)
                
                # Показываем ответ
                st.subheader("📝 Ответ:")
                st.write(answer)
                
                # Показываем источники
                st.subheader(" Источники:")
                for i, (doc, metadata) in enumerate(zip(results['documents'][0], results['metadatas'][0])):
                    source = metadata.get('source', 'Неизвестный файл')
                    st.write(f"**{i+1}. {source}**")
                    with st.expander("Показать фрагмент"):
                        st.text(doc[:500] + "..." if len(doc) > 500 else doc)
            else:
                st.warning("Не удалось найти релевантную информацию в документах.")
else:
    st.error("❌ Папка docs не найдена или пуста. Загрузите PDF-файлы в репозиторий.")
