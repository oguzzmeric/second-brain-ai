import streamlit as st
import os
import shutil
from brain2 import ask_brain_agent
from ingest2 import run_ingestion

# 1. Başlangıç temizliği
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

st.set_page_config(page_title="Second Brain AI V2", layout="wide")
st.title("Second Brain V2 - Agentic RAG 🌐")

# 2. Sidebar
with st.sidebar:
    st.header("📂 Dosya Yönetimi")
    uploaded_files = st.file_uploader("PDF dosyalarını yükle", type="pdf", accept_multiple_files=True)
    
    if st.button("Sistemi Güncelle"):
        if uploaded_files:
            with st.spinner("Dosyalar işleniyor..."):
                if not os.path.exists("data"):
                    os.makedirs("data")
                for f in uploaded_files:
                    with open(os.path.join("data", f.name), "wb") as buffer:
                        buffer.write(f.getbuffer())
                mesaj = run_ingestion()
                st.session_state.messages = []  # Eski sohbet geçmişini temizle
                st.success(mesaj)
                st.rerun()
        else:
            st.error("Lütfen önce dosya seçin.")

# 3. Sohbet Geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# 4. Kullanıcı Girişi ve Ajan Akışı
if user_input := st.chat_input("Dökümanların veya internet hakkında sorun nedir?"):
    # Kullanıcının mesajını ekrana bas ve hafızaya al
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    # Asistanın cevabı için yer ayır
    with st.chat_message("assistant"):
        full_response = ""
        message_placeholder = st.empty() 
        
        with st.status("🧠 Beyin fırtınası yapılıyor...", expanded=True) as status:
            log_placeholder = st.empty() 
            
            # Ajanı çalıştır ve gelen parçaları yakala
            for chunk in ask_brain_agent(user_input):
                
                # 1. ARA ADIMLAR (Ajan hangi tool'u kullanıyor?)
                if isinstance(chunk, dict) and "actions" in chunk:
                    for action in chunk["actions"]:
                        log_placeholder.write(f"🔍 **Karar:** `{action.tool}` kullanılıyor...")
                
                # 2. NİHAİ CEVAP
                elif isinstance(chunk, dict) and "output" in chunk:
                    output_data = chunk["output"]
                    
                    # Gemini'nin o saçma sapan liste ve imza formatını temizleme
                    if isinstance(output_data, list):
                        clean_text = ""
                        for item in output_data:
                            if isinstance(item, dict) and 'text' in item:
                                clean_text += item['text']
                            elif isinstance(item, str):
                                clean_text += item
                        full_response = clean_text
                    else:
                        full_response = str(output_data)
                        
                    # Yazıyı imleçle (▌) akıyormuş gibi göster
                    message_placeholder.markdown(full_response + "▌")
                
                # 3. YEDEK: Ajan bazen direkt string fırlatır
                elif isinstance(chunk, str):
                    full_response += chunk
                    message_placeholder.markdown(full_response + "▌")
            
            status.update(label="✅ Analiz Tamamlandı!", state="complete", expanded=False)

        # Akış bitince son kontrolleri yap ve imleci kaldır
        if full_response:
            # Ufak bir ihtimal o "CmUBDD" zırvası metnin içinde kalmışsa son kez kopar at
            if "CmUBDD" in full_response:
                full_response = full_response.split("index': 0},")[-1].strip()
                # Hala garip liste parantezleri kaldıysa temizle
                if full_response.startswith("] '"):
                    full_response = full_response[3:]
                    
            message_placeholder.markdown(full_response)
            st.session_state.messages.append({"role": "assistant", "content": full_response})
        else:
            st.error("Ajan bir cevap üretemedi. Logları kontrol et.")