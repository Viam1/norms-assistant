import os
import re
import streamlit as st
from pypdf import PdfReader
import requests

st.set_page_config(page_title="Поиск по нормам", layout="wide")

st.title("🔍 Умный помощник по нормам проектирования")
st.write("Задайте вопрос — система найдёт ответ в документах с точными цитатами.")

# 1. Проверка папки с файлами
docs_dir = "./docs"
if not os.path.exists(docs_dir):
    st.error("❌ Папка docs не найдена.")
    st.stop()

pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
if not pdf_files:
    st.error("❌ Папка docs пуста.")
    st.stop()

# 2. Боковая панель для API ключа
with st.sidebar:
    st.header("⚙️ Настройки")
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        help="Получите бесплатный ключ на console.groq.com",
        value=""
    )
    
    if not groq_api_key:
        st.warning("️ Введите API ключ Groq для работы ИИ")
        st.markdown("[Получить бесплатный ключ →](https://console.groq.com/)")
    
    st.divider()
    st.markdown("**Как это работает:**")
    st.markdown("1. Система ищет релевантные фрагменты в PDF")
    st.markdown("2. ИИ анализирует их и формулирует ответ")
    st.markdown("3. Вы получаете точный ответ с источником")

# 3. Функция чтения PDF
def get_text_from_pdf(filepath):
    pages_text = []
    try:
        reader = PdfReader(filepath)
        for page_num, page in enumerate(reader.pages, 1):
            page_text = page.extract_text()
            if page_text:
                pages_text.append({
                    "page": page_num,
                    "text": page_text
                })
    except Exception as e:
        pages_text.append({"page": 0, "text": f"[Ошибка чтения: {e}]"})
    return pages_text

# 4. Умная разбивка текста
def split_text_smart(text, chunk_size=600, overlap=150):
    chunks = []
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        if len(para) > chunk_size:
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                    current_chunk = overlap_text + " " + sentence
                else:
                    current_chunk += " " + sentence if current_chunk else sentence
        else:
            if len(current_chunk) + len(para) > chunk_size and current_chunk:
                chunks.append(current_chunk.strip())
                overlap_text = current_chunk[-overlap:] if len(current_chunk) > overlap else ""
                current_chunk = overlap_text + " " + para
            else:
                current_chunk += "\n\n" + para if current_chunk else para
    
    if current_chunk.strip():
        chunks.append(current_chunk.strip())
    
    return chunks

# 5. Загрузка и индексация
@st.cache_data(ttl=3600)
def load_and_index_all_docs():
    all_chunks = []
    
    for filename in pdf_files:
        filepath = os.path.join(docs_dir, filename)
        pages = get_text_from_pdf(filepath)
        
        for page_data in pages:
            page_num = page_data["page"]
            page_text = page_data["text"]
            
            chunks = split_text_smart(page_text, chunk_size=600, overlap=150)
            
            for chunk in chunks:
                if len(chunk) > 50:
                    all_chunks.append({
                        "text": chunk,
                        "source": filename,
                        "page": page_num
                    })
    
    return all_chunks

# 6. Поиск фрагментов
def search_chunks(chunks, query, top_k=5):
    query_words = query.lower().split()
    
    def get_word_variants(word):
        variants = [word]
        if len(word) > 4:
            variants.append(word[:-1])
            variants.append(word[:-2])
        return variants
    
    query_variants = []
    for word in query_words:
        query_variants.extend(get_word_variants(word))
    
    results = []
    for chunk in chunks:
        chunk_lower = chunk["text"].lower()
        
        score = 0
        matched_words = []
        
        for word in query_words:
            word_variants = get_word_variants(word)
            for variant in word_variants:
                if variant in chunk_lower:
                    score += 1
                    if word not in matched_words:
                        matched_words.append(word)
                    break
        
        if score > 0:
            results.append({
                "score": score,
                "chunk": chunk,
                "matched_words": matched_words
            })
    
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results[:top_k]

# 7. Получение ответа от Groq (Llama 3)
def get_ai_answer(question, context_chunks, api_key):
    if not api_key:
        return None, "API ключ не указан"
    
    # Формируем контекст из найденных фрагментов
    context = "\n\n---\n\n".join([
        f"[{chunk['source']}, стр. {chunk['page']}]\n{chunk['text']}"
        for chunk in context_chunks
    ])
    
    prompt = f"""Ты — эксперт по строительным нормам и правилам (СП, ГОСТ, СНиП). 
Используй ТОЛЬКО предоставленный контекст из документов для ответа на вопрос.
Если в контексте нет точного ответа, скажи об этом и предложи посмотреть конкретный документ.
Всегда указывай номер пункта нормы и название документа.

КОНТЕКСТ ИЗ ДОКУМЕНТОВ:
{context}

ВОПРОС: {question}

ОТВЕТ:"""
    
    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json"
    }
    
    payload = {
        "model": "llama3-70b-8192",
        "messages": [
            {"role": "system", "content": "Ты эксперт по строительным нормам. Отвечай точно по документам."},
            {"role": "user", "content": prompt}
        ],
        "temperature": 0.1,
        "max_tokens": 1024
    }
    
    try:
        response = requests.post(
            "https://api.groq.com/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        if response.status_code == 200:
            result = response.json()
            answer = result["choices"][0]["message"]["content"]
            return answer, None
        else:
            return None, f"Ошибка API: {response.status_code}"
    except Exception as e:
        return None, f"Ошибка: {str(e)}"

# 8. Основной интерфейс
with st.spinner("⏳ Индексация документов..."):
    chunks = load_and_index_all_docs()

if not chunks:
    st.warning("️ Не удалось извлечь текст из PDF.")
else:
    st.success(f"✅ Загружено **{len(chunks)}** фрагментов из **{len(pdf_files)}** документов")
    
    st.divider()
    
    # Поле ввода вопроса
    question = st.text_input(
        "💬 Задайте вопрос:",
        placeholder="Например: Сколько метров должно быть до пожарного выхода? Какая высота потолков в стоматологии?",
        key="question_box",
        help="Система найдёт релевантные фрагменты и ИИ сформулирует точный ответ"
    )
    
    if question:
        if not groq_api_key:
            st.error("⚠️ Сначала введите API ключ Groq в боковой панели")
        else:
            with st.spinner(" Поиск в документах..."):
                # Шаг 1: Поиск релевантных фрагментов
                relevant_chunks = search_chunks(chunks, question, top_k=5)
                
                if not relevant_chunks:
                    st.warning("🔍 Не найдено релевантных фрагментов. Попробуйте переформулировать вопрос.")
                else:
                    st.info(f"📄 Найдено {len(relevant_chunks)} релевантных фрагментов")
                    
                    # Шаг 2: Получение ответа от ИИ
                    with st.spinner("🤖 ИИ формулирует ответ..."):
                        answer, error = get_ai_answer(question, relevant_chunks, groq_api_key)
                    
                    if error:
                        st.error(f"❌ {error}")
                    else:
                        # Показываем ответ от ИИ
                        st.subheader("📝 Ответ:")
                        st.markdown(answer)
                        
                        st.divider()
                        
                        # Показываем источники
                        st.subheader("📚 Источники:")
                        for i, result in enumerate(relevant_chunks, 1):
                            chunk = result["chunk"]
                            with st.expander(f"**{i}. {chunk['source']}** (стр. {chunk['page']})"):
                                st.text(chunk["text"])
