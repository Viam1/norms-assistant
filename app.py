import os
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Поиск по нормам", layout="wide")

st.title(" Умный поиск по нормам проектирования")
st.write("Введите ключевые слова — система найдёт точные цитаты из документов.")

# 1. Проверка папки с файлами
docs_dir = "./docs"
if not os.path.exists(docs_dir):
    st.error("❌ Папка docs не найдена.")
    st.stop()

pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
if not pdf_files:
    st.error("❌ Папка docs пуста.")
    st.stop()

# 2. Функция чтения PDF (упрощённая и быстрая)
def get_text_from_pdf(filepath):
    text = ""
    try:
        reader = PdfReader(filepath)
        for page in reader.pages:
            page_text = page.extract_text()
            if page_text:
                text += page_text + "\n"
    except Exception as e:
        text += f"\n[Ошибка чтения файла: {e}]\n"
    return text

# 3. Загрузка и разбивка на фрагменты (выполняется один раз)
@st.cache_data(ttl=3600) # Кэшируем на 1 час, чтобы не зависало
def load_and_split_all_docs():
    all_chunks = []
    # Берём все файлы, но ограничиваем размер для скорости
    for filename in pdf_files:
        filepath = os.path.join(docs_dir, filename)
        full_text = get_text_from_pdf(filepath)
        
        # Разбиваем текст на абзацы/предложения (примерно по 500-800 знаков)
        # Используем разбивку по двойному переносу строки или просто по длине
        paragraphs = full_text.split('\n\n')
        
        for para in paragraphs:
            clean_para = para.strip()
            if 50 < len(clean_para) < 1500: # Берём только осмысленные куски
                all_chunks.append({
                    "text": clean_para,
                    "source": filename
                })
    return all_chunks

# 4. Основной интерфейс
with st.spinner("⏳ Индексация документов (это займёт пару секунд)..."):
    chunks = load_and_split_all_docs()

if not chunks:
    st.warning("Не удалось извлечь текст из PDF. Возможно, это сканы.")
else:
    st.success(f"✅ Готово! Проиндексировано {len(chunks)} фрагментов из {len(pdf_files)} файлов.")
    
    # Поле поиска
    query = st.text_input("🔎 Что ищем?", placeholder="Например: высота потолков, вентиляция, освещение", key="search_box")
    
    if query:
        query_lower = query.lower()
        # Разбиваем запрос на слова для более точного поиска
        search_words = query_lower.split()
        
        # Ищем фрагменты, где есть хотя бы одно слово из запроса
        results = []
        for chunk in chunks:
            chunk_lower = chunk["text"].lower()
            # Считаем, сколько слов из запроса найдено в фрагменте
            score = sum(1 for word in search_words if word in chunk_lower)
            if score > 0:
                results.append((score, chunk))
        
        # Сортируем: сначала те, где больше совпадений
        results.sort(key=lambda x: x[0], reverse=True)
        
        # Показываем топ-10 результатов
        st.subheader(f"Найдено результатов: {len(results)}")
        
        if len(results) == 0:
            st.info("Ничего не найдено. Попробуйте другие слова.")
        else:
            for score, chunk in results[:10]:
                with st.container():
                    st.markdown(f"** Источник:** `{chunk['source']}`")
                    st.markdown(f"**Текст:** {chunk['text']}")
                    st.markdown("---")
