import pandas as pd
from datasets import Dataset
from ragas import evaluate
from ragas.run_config import RunConfig  
from ragas.metrics import ContextRecall, AnswerRelevancy

from langchain_ollama import ChatOllama
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma
from brain2 import ask_brain 

print("🚀 RAGAS Lokal Değerlendirme Sistemi Başlatılıyor...")

# 1. Modelleri Tanımla
eval_llm = ChatOllama(model="llama3", temperature=0)
eval_embeddings = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

# 2. Veritabanını Yükle
print("📂 Diskten (chroma_db) vektör veritabanı yükleniyor...")
vectordb = Chroma(persist_directory="chroma_db", embedding_function=eval_embeddings)
retriever = vectordb.as_retriever(search_kwargs={"k": 5}) # Hız için k'yı 5 yaptık

# 3. TEKNOFEST 2026 Şartnamesi Test Verileri[cite: 1]
sorular = [
    "Birinci görev kapsamında tespit edilmesi beklenen nesne türleri (sınıfları) nelerdir?",
    "Yarışma oturumlarında verilecek videoların saniyedeki kare sayısı (FPS) ne kadardır?",
    "Uçan Araba Park (UAP) ve Uçan Ambulans İniş (UAİ) alanlarının çapı kaç metredir?",
    "Genel yarışma puanlandırmasında İkinci Görevin puan ağırlığı yüzde kaçtır?"
]

beklenen_cevaplar = [
    "Nesne türleri taşıt, insan, Uçan Araba Park (UAP) ve Uçan Ambulans İniş (UAİ) alanları olmak üzere 4 adettir.[cite: 1]",
    "Saniyedeki kare sayısı (FPS) 7.5 olacaktır.[cite: 1]",
    "UAP ve UAİ alanları 4,5 metre çapındadır.[cite: 1]",
    "İkinci görev puan oranı %40'tır.[cite: 1]"
]

# 4. Yanıtları Topla
sistem_cevaplari = []
sistem_baglamlari = []

print("⚙️ Senin RAG sistemin test ediliyor, lütfen bekle...")
for soru in sorular:
    docs = retriever.invoke(f"query: {soru}")
    sistem_baglamlari.append([doc.page_content for doc in docs])
    sistem_cevaplari.append(ask_brain(soru))

dataset = Dataset.from_dict({
    "question": sorular,
    "answer": sistem_cevaplari,
    "contexts": sistem_baglamlari,
    "ground_truth": beklenen_cevaplar
})

# 5. Değerlendirme (Hızlı ve Güvenli Mod)
lokal_run_config = RunConfig(timeout=300, max_workers=1)

print("🧠 RAGAS Puanları Hesaplanıyor...")
sonuc = evaluate(
    dataset=dataset,
    metrics=[ContextRecall(), AnswerRelevancy()],
    llm=eval_llm,
    embeddings=eval_embeddings,
    run_config=lokal_run_config
)

# 6. SONUÇLARI BAS
print("\n" + "="*50)
print("🎯 TEKNOFEST RAG ANALİZ SONUÇLARI:")
print(sonuc) 
print("="*50)