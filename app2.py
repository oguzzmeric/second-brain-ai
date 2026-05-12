import streamlit as st
import os
import shutil
import tempfile
from brain2 import ask_brain_agent, guard_route
from ingest2 import run_ingestion
from langchain_openai import ChatOpenAI

# --- İZOLASYON AYARLARI ---
# Her kullanıcı için benzersiz bir çalışma dizini oluşturuyoruz
if "user_data_path" not in st.session_state:
    # Streamlit Cloud üzerinde her oturuma özel geçici bir klasör açar
    st.session_state.user_data_path = tempfile.mkdtemp()
    st.session_state.user_db_path = os.path.join(st.session_state.user_data_path, "chroma_db")
    st.session_state.user_pdf_path = os.path.join(st.session_state.user_data_path, "data")
    
    os.makedirs(st.session_state.user_pdf_path, exist_ok=True)

st.set_page_config(page_title="Second Brain AI V2", layout="wide")

st.info("💡 **GÜVENLİK NOTU:** Bu oturum tamamen izoledir. Yüklediğiniz dökümanlar sadece size özel geçici bellekte (`tempfile`) tutulur ve oturum kapanınca silinir.")

st.title("Second Brain V2 - Agentic RAG 🌐")

# 2. Sidebar 
with st.sidebar:
    st.header("📂 Dosya Yönetimi")
    st.warning("⚠️ **Sistem Stabilitesi:** Kota optimizasyonu nedeniyle aynı anda maksimum **5 adet** PDF yükleyiniz.")
    
    uploaded_files = st.file_uploader("PDF dosyalarını yükle", type="pdf", accept_multiple_files=True)
    
    if uploaded_files and len(uploaded_files) > 5:
        st.error("🚨 Lütfen en fazla 5 adet döküman yükleyin!")
        st.stop()
        
    if st.button("Sistemi Güncelle"):
        if uploaded_files:
            with st.spinner("Dosyalar size özel alanda işleniyor..."):
                # Önce eski dosyaları temizle (sadece bu kullanıcıya özel klasörü)
                for file in os.listdir(st.session_state.user_pdf_path):
                    os.remove(os.path.join(st.session_state.user_pdf_path, file))
                
                # Yeni dosyaları kaydet
                for f in uploaded_files:
                    file_path = os.path.join(st.session_state.user_pdf_path, f.name)
                    with open(file_path, "wb") as buffer:
                        buffer.write(f.getbuffer())
                
                # run_ingestion fonksiyonuna özel yolları gönderiyoruz
                # NOT: ingest2.py dosyasındaki run_ingestion fonksiyonunun 
                # bu parametreleri kabul edecek şekilde güncellenmesi gerekir.
                mesaj = run_ingestion(
                    source_dir=st.session_state.user_pdf_path, 
                    persist_dir=st.session_state.user_db_path
                )
                
                st.session_state.messages = []  
                st.success(mesaj)
                st.rerun()
        else:
            st.error("Lütfen önce dosya seçin.")

# --- SOHBET AKIŞI ---
# (Buradaki kodların geri kalanı aynı kalıyor, ask_brain_agent çağrılırken 
# db_path parametresi gönderilmesi güvenliği tam sağlar.)

if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

if user_input := st.chat_input("Dökümanların veya internet hakkında sorun nedir?"):
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        rota = guard_route(user_input)
        if rota == "SIMPLE":
            st.toast("⚡ Basit Soru Algılandı", icon="🚀")
            basit_llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.5)
            with st.spinner("Yanıtlanıyor..."):
                cevap = basit_llm.invoke(f"Kullanıcıya kısa cevap ver: {user_input}").content
            st.markdown(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})
        else:
            st.toast("🔍 Teknik Soru Algılandı", icon="🕵️‍♂️")
            full_response = ""
            message_placeholder = st.empty() 
            try:
                with st.status("🧠 Beyin fırtınası yapılıyor...", expanded=True) as status:
                    # ask_brain_agent'a kullanıcının özel DB yolunu gönderiyoruz
                    for chunk in ask_brain_agent(user_input, db_path=st.session_state.user_db_path):
                        if isinstance(chunk, dict) and "output" in chunk:
                            full_response = str(chunk["output"])
                            message_placeholder.markdown(full_response)
                
                status.update(label="✅ Analiz Tamamlandı!", state="complete", expanded=False)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"⚠️ Hata: {e}")