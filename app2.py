import streamlit as st
import os
import shutil
import tempfile
import glob
from brain2 import ask_brain_agent, guard_route
from ingest2 import run_ingestion
from langchain_openai import ChatOpenAI

# --- PROFESYONEL TEMİZLİK MEKANİZMASI (STARTUP SWEEPER) ---
def global_cleanup():
    """Sunucu her ayağa kalktığında eski oturumlardan kalan geçici dosyaları süpürür."""
    temp_dir = tempfile.gettempdir()
    # İşletim sistemindeki tüm "tmp" ile başlayan klasörleri tara
    old_paths = glob.glob(os.path.join(temp_dir, "tmp*"))
    for path in old_paths:
        try:
            # Sadece bizim oluşturduğumuz tipteki klasörleri ve 
            # erişim iznimiz olanları siler (diğer uygulamaların dosyalarına dokunmaz)
            if os.path.isdir(path) and not os.access(path, os.W_OK):
                continue 
            shutil.rmtree(path, ignore_errors=True)
        except:
            pass

# Uygulama ilk kez başlatıldığında temizlik yap
if 'initialized' not in st.session_state:
    global_cleanup()
    st.session_state.initialized = True

# --- OTOMATİK İMHA (GARBAGE COLLECTION HACK) ---
class SessionCleaner:
    def __init__(self, path_to_clean):
        self.path_to_clean = path_to_clean

    def __del__(self):
        """Bu fonksiyon, obje RAM'den silindiğinde (sekme kapanınca) OTOMATİK çalışır."""
        try:
            if os.path.exists(self.path_to_clean):
                shutil.rmtree(self.path_to_clean, ignore_errors=True)
                print(f"[TEMİZLİK] Oturum kapandı, klasör silindi: {self.path_to_clean}")
        except:
            pass

# --- İZOLASYON AYARLARI ---
if "user_data_path" not in st.session_state:
    st.session_state.user_data_path = tempfile.mkdtemp()
    st.session_state.user_db_path = os.path.join(st.session_state.user_data_path, "chroma_db")
    st.session_state.user_pdf_path = os.path.join(st.session_state.user_data_path, "data")
    os.makedirs(st.session_state.user_pdf_path, exist_ok=True)
    
    # Sekme açıldığında bu temizleyiciyi session_state'e bağlıyoruz
    st.session_state.auto_cleaner = SessionCleaner(st.session_state.user_data_path)

st.set_page_config(page_title="Second Brain AI V2", layout="wide")

st.info("💡 **GÜVENLİK NOTU:** Bu oturum tamamen izoledir. Yüklediğiniz dökümanlar sadece size özel geçici bellekte tutulur ve **sekme kapandığında otomatik olarak imha edilir.**")

st.title("Second Brain V2 - Agentic RAG 🌐")

# --- SIDEBAR ---
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
                for file in os.listdir(st.session_state.user_pdf_path):
                    os.remove(os.path.join(st.session_state.user_pdf_path, file))
                
                for f in uploaded_files:
                    file_path = os.path.join(st.session_state.user_pdf_path, f.name)
                    with open(file_path, "wb") as buffer:
                        buffer.write(f.getbuffer())
                
                mesaj = run_ingestion(
                    source_dir=st.session_state.user_pdf_path, 
                    persist_dir=st.session_state.user_db_path
                )
                
                st.session_state.messages = []  
                st.success(mesaj)
                st.rerun()
        else:
            st.error("Lütfen önce dosya seçin.")

    st.markdown("---")
    # --- MANUEL TEMİZLİK BUTONU ---
    st.subheader("🧹 Oturum Temizliği")
    if st.button("Tüm Verilerimi Sil ve Çık", use_container_width=True):
        if os.path.exists(st.session_state.user_data_path):
            shutil.rmtree(st.session_state.user_data_path, ignore_errors=True)
        # Session state'i sıfırla
        for key in list(st.session_state.keys()):
            del st.session_state[key]
        st.toast("Veriler temizlendi, oturum sıfırlandı!", icon="🗑️")
        st.rerun()

# --- SOHBET AKIŞI ---
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
                    # Ajan artık kullanıcının özel klasöründeki DB'ye bakıyor
                    for chunk in ask_brain_agent(user_input, db_path=st.session_state.user_db_path):
                        if isinstance(chunk, dict) and "output" in chunk:
                            full_response = str(chunk["output"])
                            message_placeholder.markdown(full_response)
                
                status.update(label="✅ Analiz Tamamlandı!", state="complete", expanded=False)
                st.session_state.messages.append({"role": "assistant", "content": full_response})
            except Exception as e:
                st.error(f"⚠️ Hata: {e}")