import streamlit as st
import os
import shutil
from brain2 import ask_brain
from ingest2 import run_ingestion

st.set_page_config(page_title="Second Brain AI", layout="wide")
st.title("🧠 İkinci Beyin: Teknik Analiz Paneli")

# YAN PANEL: DOSYA YÜKLEME
with st.sidebar:
    st.header("📂 Dosya Yönetimi")
    uploaded_files = st.file_uploader("PDF dosyalarını yükle", type="pdf", accept_multiple_files=True)
    
    if st.button("Sistemi Güncelle"):
        if uploaded_files:
            with st.spinner("Dosyalar işleniyor..."):
                # Önce 'data' klasörünü kontrol et ve temizle
                if os.path.exists("data"):
                    shutil.rmtree("data")
                os.makedirs("data")
                
                # Dosyaları site üzerinden klasöre kaydet
                for f in uploaded_files:
                    with open(os.path.join("data", f.name), "wb") as buffer:
                        buffer.write(f.getbuffer())
                
                # Ingest motorunu çalıştır
                mesaj = run_ingestion()
                st.success(mesaj)
                st.rerun()
        else:
            st.error("Lütfen önce dosya seçin.")

# SOHBET EKRANI
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
        with st.spinner("Cevap aranıyor..."):
            if os.path.exists("chroma_db"):
                response = ask_brain(user_input)
                st.markdown(response)
                st.session_state.messages.append({"role": "assistant", "content": response})
            else:
                st.error("Veritabanı bulunamadı. Lütfen önce döküman yükleyip sistemi güncelleyin.")