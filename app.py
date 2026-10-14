import os
import re
import streamlit as st
from pypdf import PdfReader

st.set_page_config(page_title="Поиск по нормам", layout="wide")

st.title("🔍 Умный поиск по нормам проектирования")
st.write("Введите запрос — система найдёт релевантные цитаты из документов.")

# 1. Проверка папки с файлами
docs_dir = "./docs"
if not os.path.exists(docs_dir):
    st.error("❌ Папка docs не найдена.")
    st.stop()

pdf_files = [f for f in os.listdir(docs_dir) if f.endswith('.pdf')]
if not pdf_files:
    st.error("❌ Папка docs пуста.")
    st.stop()

# 2. Функция чтения PDF с извлечением текста постранично
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

# 3. Умная разбивка текста на фрагменты
def split_text_smart(text, chunk_size=500, overlap=100):
    """Разбивает текст на фрагменты с перекрытием, стараясь не разрывать предложения"""
    chunks = []
    
    # Сначала разбиваем по абзацам (двойной перенос строки)
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    for para in paragraphs:
        para = para.strip()
        if not para:
            continue
        
        # Если абзац очень длинный, разбиваем его по предложениям
        if len(para) > chunk_size:
            # Разбиваем по точкам с пробелом
            sentences = re.split(r'(?<=[.!?])\s+', para)
            for sentence in sentences:
                sentence = sentence.strip()
                if not sentence:
                    continue
                
                if len(current_chunk) + len(sentence) > chunk_size and current_chunk:
                    chunks.append(current_chunk.strip())
                    # Оставляем перекрытие
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

# 4. Загрузка и индексация всех документов
@st.cache_data(ttl=3600)
def load_and_index_all_docs():
    all_chunks = []
    
    for filename in pdf_files:
        filepath = os.path.join(docs_dir, filename)
        pages = get_text_from_pdf(filepath)
        
        for page_data in pages:
            page_num = page_data["page"]
            page_text = page_data["text"]
            
            # Разбиваем страницу на фрагменты
            chunks = split_text_smart(page_text, chunk_size=600, overlap=150)
            
            for chunk in chunks:
                if len(chunk) > 50:  # Пропускаем слишком короткие
                    all_chunks.append({
                        "text": chunk,
                        "source": filename,
                        "page": page_num
                    })
    
    return all_chunks

# 5. Улучшенный поиск с морфологией и подсветкой
def search_chunks(chunks, query):
    """Ищет фрагменты с учётом морфологии и частичного совпадения"""
    # Разбиваем запрос на слова
    query_words = query.lower().split()
    
    # Для каждого слова создаём варианты (упрощённая морфология)
    def get_word_variants(word):
        variants = [word]
        # Убираем окончания (очень упрощённо)
        if len(word) > 4:
            variants.append(word[:-1])  # убираем последнюю букву
            variants.append(word[:-2])  # убираем две последние
        return variants
    
    query_variants = []
    for word in query_words:
        query_variants.extend(get_word_variants(word))
    
    results = []
    for chunk in chunks:
        chunk_lower = chunk["text"].lower()
        
        # Считаем количество совпадений
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
    
    # Сортируем по количеству совпадений
    results.sort(key=lambda x: x["score"], reverse=True)
    
    return results

# 6. Подсветка найденных слов в тексте
def highlight_text(text, words):
    """Подсвечивает найденные слова в тексте"""
    highlighted = text
    for word in words:
        # Ищем слово в разных формах (упрощённо)
        pattern = re.compile(re.escape(word[:4]) + r'\w*', re.IGNORECASE)
        highlighted = pattern.sub(lambda m: f"**{m.group()}**", highlighted)
    return highlighted

# 7. Основной интерфейс
with st.spinner("⏳ Индексация документов..."):
    chunks = load_and_index_all_docs()

if not chunks:
    st.warning("️ Не удалось извлечь текст из PDF. Возможно, это сканы.")
else:
    st.success(f"✅ Готово! Проиндексировано **{len(chunks)}** фрагментов из **{len(pdf_files)}** файлов.")
    
    # Поле поиска
    query = st.text_input(
        "🔎 Что ищем?", 
        placeholder="Например: пожарный выход, высота потолков, вентиляция кухня",
        key="search_box",
        help="Можно вводить несколько слов. Система найдёт фрагменты с наибольшим количеством совпадений."
    )
    
    if query:
        results = search_chunks(chunks, query)
        
        st.subheader(f"📊 Найдено результатов: {len(results)}")
        
        if len(results) == 0:
            st.info("🔍 Ничего не найдено. Попробуйте:")
            st.markdown("""
            - Использовать другие слова или синонимы
            - Вводить запрос короче (2-3 слова)
            - Проверить орфографию
            """)
        else:
            # Показываем топ-20 результатов
            st.markdown(f"Показаны лучшие **{min(20, len(results))}** результатов:")
            
            for i, result in enumerate(results[:20], 1):
                chunk = result["chunk"]
                matched_words = result["matched_words"]
                
                with st.container():
                    # Заголовок с источником
                    st.markdown(f"**{i}. 📄 `{chunk['source']}` (стр. {chunk['page']})**")
                    
                    # Текст с подсветкой
                    highlighted_text = highlight_text(chunk["text"], matched_words)
                    st.markdown(highlighted_text)
                    
                    # Найденные слова
                    if matched_words:
                        st.markdown(f"🏷️ Найдено слов: {', '.join(matched_words)}")
                    
                    st.markdown("---")
