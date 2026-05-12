from langchain_openai import ChatOpenAI
import streamlit as st
import os
import shutil
from brain2 import ask_brain_agent, guard_route
from ingest2 import run_ingestion
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_groq import ChatGroq

#  temizlik
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

# -
st.info("💡 **MİMARİ NOT:** Bu canlı demo, erişim kolaylığı için Serverless API (Gemini) ve Hibrit Yönlendirme (Semantic Routing) ile derlenmiştir. Kurumsal gizlilik gereksinimlerinde saniyeler içinde izole yerel modellere (Air-Gapped Private Cloud) geçiş yapılabilir.")

st.title("Second Brain V2 - Agentic RAG 🌐")

# 2. Sidebar 
with st.sidebar:
    st.header("📂 Dosya Yönetimi")
    st.warning("⚠️ **Sistem Stabilitesi:** Kota optimizasyonu nedeniyle aynı anda maksimum **5 adet** PDF yükleyiniz.")
    
    uploaded_files = st.file_uploader("PDF dosyalarını yükle", type="pdf", accept_multiple_files=True)
    
    # 5 sınırı koydum 
    if uploaded_files and len(uploaded_files) > 5:
        st.error("🚨 Lütfen en fazla 5 adet döküman yükleyin!")
        st.stop()
        
    if st.button("Sistemi Güncelle"):
        if uploaded_files:
            with st.spinner("Dosyalar işleniyor..."):
                if not os.path.exists("data"):
                    os.makedirs("data")
                for f in uploaded_files:
                    with open(os.path.join("data", f.name), "wb") as buffer:
                        buffer.write(f.getbuffer())
                mesaj = run_ingestion()
                st.session_state.messages = []  
                st.success(mesaj)
                st.rerun()
        else:
            st.error("Lütfen önce dosya seçin.")

# Sohbet Geçmişi
if "messages" not in st.session_state:
    st.session_state.messages = []

for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

#  HİBRİT Akış
if user_input := st.chat_input("Dökümanların veya internet hakkında sorun nedir?"):
    
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.markdown(user_input)

    with st.chat_message("assistant"):
        rota = guard_route(user_input)
        if rota == "SIMPLE":
            
            st.toast("⚡ Basit Soru Algılandı: Hızlı model yanıtlıyor...", icon="🚀")
            basit_llm = ChatOpenAI(
                model="gpt-3.5-turbo", #kota için minimal bir model seçtim
                temperature=0.5
                )
            
            with st.spinner("Yanıtlanıyor..."):
                cevap = basit_llm.invoke(f"Sen Oğuz'un tasarladığı akıllı asistanısın. Kullanıcıya kısa ve net cevap ver: {user_input}").content
            
            st.markdown(cevap)
            st.session_state.messages.append({"role": "assistant", "content": cevap})
            
        else:
            # KOMPLEKS SORULAR İÇİN AGENT AKIŞI
            st.toast("🔍 Teknik Soru Algılandı: RAG Ajanı devrede...", icon="🕵️‍♂️")
            full_response = ""
            message_placeholder = st.empty() 
            
            try:
                with st.status("🧠 Beyin fırtınası yapılıyor...", expanded=True) as status:
                    log_placeholder = st.empty() 
                    
                    for chunk in ask_brain_agent(user_input):
                        
                        if isinstance(chunk, dict) and "actions" in chunk:
                            for action in chunk["actions"]:
                                log_placeholder.write(f"🔍 **Karar:** `{action.tool}` kullanılıyor...")
                        
                        #CEVAP
                        elif isinstance(chunk, dict) and "output" in chunk:
                            output_data = chunk["output"]
                            
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
                            message_placeholder.markdown(full_response + "▌")
                        elif isinstance(chunk, str):
                            full_response += chunk
                            message_placeholder.markdown(full_response + "▌")
                    
                    status.update(label="✅ Analiz Tamamlandı!", state="complete", expanded=False)

                #  CmUBDD ile başlaya gereksiz verilerden kurtulmak için kontrol ekledim
                if full_response:
                    if "CmUBDD" in full_response:
                        full_response = full_response.split("index': 0},")[-1].strip()
                        if full_response.startswith("] '"):
                            full_response = full_response[3:]
                            
                    message_placeholder.markdown(full_response)
                    st.session_state.messages.append({"role": "assistant", "content": full_response})
                else:
                    st.error("Ajan bir cevap üretemedi. Logları kontrol et.")
                    
            except Exception as e:
                # KOTA (429) VE HATA YAKALAYICI
                if "429" in str(e) or "ResourceExhausted" in str(e):
                    hata_mesaji = "⏳ **Sistem Yoğunluğu:** Şu an mülakat değerlendirmeleri nedeniyle aşırı trafik alıyoruz. Google API sınırına ulaşıldı. Lütfen **15 saniye sonra** tekrar deneyin."
                    st.error(hata_mesaji)
                    st.session_state.messages.append({"role": "assistant", "content": hata_mesaji})
                else:
                    st.error(f"⚠️ Ajan çalışırken bir hata oluştu: {e}")