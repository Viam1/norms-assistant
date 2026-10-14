import os
import streamlit as st
from pypdf import PdfReader
import numpy as np
import requests

st.set_page_config(page_title="Помощник по нормам", layout="wide")

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

# Простой поиск по ключевым словам (без векторной БД — надёжнее)
def search_chunks(chunks_data, query, top_k=3):
    query_words = set(query.lower().split())
    scored = []
    for chunk in chunks_data:
        text_lower = chunk['text'].lower()
        score = sum(1 for word in query_words if word in text_lower)
        if score > 0:
            scored.append((score, chunk))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [item[1] for item in scored[:top_k]]

# Загрузка документов
@st.cache_data
def load_documents():
    docs_dir = "./docs"
    
    if not os.path.exists(docs_dir):
        return None
    
    pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
    
    if not pdf_files:
        return None
    
    all_chunks = []
    
    for filename in pdf_files:
        pdf_path = os.path.join(docs_dir, filename)
        text = extract_text_from_pdf(pdf_path)
        
        if not text.strip():
            continue
            
        chunks = split_text(text)
        
        for i, chunk in enumerate(chunks):
            all_chunks.append({
                "text": chunk,
                "source": filename,
                "chunk_id": i
            })
    
    return all_chunks

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
with st.spinner("Загрузка документов..."):
    chunks_data = load_documents()

if chunks_data:
    st.success(f"✅ Загружено {len(chunks_data)} фрагментов из документов.")
    
    # Поле ввода
    question = st.text_input("Ваш вопрос:", placeholder="Например: Какие требования к высоте потолков?")
    
    if question:
        with st.spinner("Поиск ответа в документах..."):
            # Ищем релевантные фрагменты
            relevant_chunks = search_chunks(chunks_data, question, top_k=3)
            
            if relevant_chunks:
                # Формируем контекст
                context = "\n\n---\n\n".join([chunk['text'] for chunk in relevant_chunks])
                
                # Получаем ответ от модели
                answer = get_answer_from_model(context, question)
                
                # Показываем ответ
                st.subheader(" Ответ:")
                st.write(answer)
                
                # Показываем источники
                st.subheader("📚 Источники:")
                for i, chunk in enumerate(relevant_chunks):
                    source = chunk.get('source', 'Неизвестный файл')
                    st.write(f"**{i+1}. {source}**")
                    with st.expander("Показать фрагмент"):
                        st.text(chunk['text'][:500] + "..." if len(chunk['text']) > 500 else chunk['text'])
            else:
                st.warning("Не удалось найти релевантную информацию в документах.")
else:
    st.error("❌ Папка docs не найдена или пуста. Загрузите PDF-файлы в репозиторий.")
