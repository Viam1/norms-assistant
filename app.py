import os
import streamlit as st
from pypdf import PdfReader
import requests

st.set_page_config(page_title="Помощник по нормам", layout="wide")

st.title(" Помощник по нормам проектирования")
st.write("Задайте вопрос — ответ будет найден в загруженных документах.")

# Проверка папки docs
docs_dir = "./docs"
if not os.path.exists(docs_dir):
    st.error("❌ Папка docs не найдена. Загрузите PDF-файлы в репозиторий.")
    st.stop()

pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
if not pdf_files:
    st.error("❌ Папка docs пуста. Загрузите PDF-файлы.")
    st.stop()

st.info(f"📁 Найдено PDF-файлов: {len(pdf_files)}")

# Функция извлечения текста
def extract_text_from_pdf(pdf_path):
    try:
        reader = PdfReader(pdf_path)
        text = ""
        for i, page in enumerate(reader.pages):
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
        return text
    except Exception as e:
        return f"Ошибка чтения файла: {str(e)}"

# Загружаем только первые 3 PDF для теста
st.info("⏳ Загружаю первые 3 документа для теста...")

test_files = pdf_files[:3]
all_text = ""

for filename in test_files:
    pdf_path = os.path.join(docs_dir, filename)
    text = extract_text_from_pdf(pdf_path)
    all_text += f"\n\n=== {filename} ===\n\n{text}"
    st.text(f"✅ Загружен: {filename}")

st.success(f"✅ Загружено {len(test_files)} документов. Общий объём текста: {len(all_text)} символов.")

# Поле ввода
question = st.text_input("Ваш вопрос:", placeholder="Например: Какие требования к высоте потолков?")

if question:
    with st.spinner("Поиск ответа..."):
        # Простой поиск по ключевым словам
        query_words = set(question.lower().split())
        
        # Разбиваем текст на фрагменты по 500 символов
        chunks = []
        for i in range(0, len(all_text), 500):
            chunk = all_text[i:i+500]
            if chunk.strip():
                chunks.append(chunk)
        
        # Ищем релевантные фрагменты
        scored_chunks = []
        for chunk in chunks:
            chunk_lower = chunk.lower()
            score = sum(1 for word in query_words if word in chunk_lower)
            if score > 0:
                scored_chunks.append((score, chunk))
        
        scored_chunks.sort(key=lambda x: x[0], reverse=True)
        top_chunks = [item[1] for item in scored_chunks[:3]]
        
        if top_chunks:
            context = "\n\n---\n\n".join(top_chunks)
            
            # Отправляем запрос к модели
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
                    answer = result[0].get('generated_text', 'Не удалось получить ответ')
                else:
                    answer = "Модель временно недоступна. Попробуйте через минуту."
            except Exception as e:
                answer = f"Ошибка: {str(e)}"
            
            st.subheader("📝 Ответ:")
            st.write(answer)
            
            st.subheader(" Найденные фрагменты:")
            for i, chunk in enumerate(top_chunks):
                st.write(f"**Фрагмент {i+1}:**")
                st.text(chunk[:300] + "..." if len(chunk) > 300 else chunk)
        else:
            st.warning("Не удалось найти релевантную информацию.")
