import streamlit as st
import os
import shutil
from brain2 import ask_brain
from ingest2 import run_ingestion

# clear datasets and db on app start
def initial_cleanup():
    paths_to_clean = ["chroma_db", "data"]
    for path in paths_to_clean:
        if os.path.exists(path):
            try:
                shutil.rmtree(path)
                print(f"[SİSTEM] Eski {path} temizlendi.")
            except Exception as e:
                print(f"[HATA] {path} silinemedi: {e}")


if "cleaned" not in st.session_state:
    initial_cleanup()
    st.session_state.cleaned = True

st.set_page_config(page_title="Second Brain AI", layout="wide")
st.title("🧠 İkinci Beyin: Teknik Analiz Paneli")

# side bar for file upload and ingestion
with st.sidebar:
    st.header("📂 Dosya Yönetimi")
    uploaded_files = st.file_uploader("PDF dosyalarını yükle", type="pdf", accept_multiple_files=True)
    
    if st.button("Sistemi Güncelle"):
        if uploaded_files:
            with st.spinner("Dosyalar işleniyor..."):
                # Data klasörünü hazırla
                if not os.path.exists("data"):
                    os.makedirs("data")
                
                for f in uploaded_files:
                    with open(os.path.join("data", f.name), "wb") as buffer:
                        buffer.write(f.getbuffer())
                
                # Ingest motorunu çalıştır
                mesaj = run_ingestion()
                st.success(mesaj)
                st.rerun()
        else:
            st.error("Lütfen önce dosya seçin.")

# Chat screen
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Döküman hakkında sorun nedir?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        with st.spinner(" düşünüyorum ..."):
            if os.path.exists("chroma_db"):
                response_stream = ask_brain(user_input)
                full_response = st.write_stream(response_stream)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            else:
                st.warning("Henüz bir döküman yüklenmedi. Lütfen önce PDF dosyalarını yükleyin ve 'Sistemi Güncelle' butonuna basın.")

