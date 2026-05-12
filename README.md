# 🧠 Second Brain AI V2: Agentic RAG Ecosystem

**Second Brain AI V2**, standart bir döküman sorgulama sisteminin ötesine geçerek; **Otonom Ajan (Agentic)** mimarisi ve **Oturum Tabanlı Veri İzolasyonu** ile geliştirilmiş ileri seviye bir RAG sistemidir. 

---

## 🚀 v2 ile Gelen Yenilikler (Evolution)

* **Agentic Workflow:** LangChain "Tool Calling" ajanı sayesinde sistem, soruyu analiz ederek Internal PDF Search mi yoksa Web Search mü yapacağına kendisi karar verir.

* **Multi-user Isolation (Zero-Footprint):** tempfile ve Session State entegrasyonu ile her kullanıcının verisi tamamen izole edilir.

* **Hybrid Intelligence:** Kompleks analizler için **GPT-4o-mini**, basit sohbetler için **GPT-3.5-Turbo** kullanılarak maliyet optimizasyonu sağlanmıştır.

* **Cross-Document Reasoning:** Geliştirilmiş $k = 20$ parametresi ile farklı dökümanlardaki bilgiler birleştirilir.

* **Cloud Native Deployment:** Streamlit Cloud üzerinde CI/CD süreçleriyle tamamen canlıya alınmıştır.

---

## 🛠️ Teknik Yığın (Tech Stack)

| Katman | Teknoloji |
| :--- | :--- |
| **Zeka (LLM)** | OpenAI GPT-4o-mini & GPT-3.5-Turbo |
| **Orkestrasyon** | LangChain (Agent & Tool Calling) |
| **Vektör Veritabanı** | ChromaDB (Session-based) |
| **Embedding** | HuggingFace intfloat/multilingual-e5-base |
| **Arayüz** | Streamlit (Custom Logic) |

---

## 📐 Mimari Akış

1. **Semantic Routing:** Giriş bir "Bekçi" LLM tarafından analiz edilir.
2. **Context Retrieval:** Teknik sorularda ajana özel `db_path` üzerinden tarama yapılır.
3. **Web Augmentation:** Eksik bilgiler internet araçlarıyla tamamlanır.
4. **Synthesis:** Çapraz kontrol edilerek son yanıt üretilir.

---

## 💻 Kurulum

### 1. Çevresel Değişkenler
`.env` dosyasına anahtarınızı ekleyin:
`OPENAI_API_KEY=your_api_key`

### 2. Bağımlılıklar
`pip install -r requirements.txt`

### 3. Çalıştırma
`streamlit run app2.py`

---

## 📂 Proje Yapısı

- **app2.py**: Main Entry. Oturum izolasyonu ve UI yönetimi.
- **brain2.py**: Logic Layer. Ajan ve tool tanımlamaları.
- **ingest2.py**: Data Engine. PDF parçalama ve vektör inşası.

---
-Oğuz Umut Meriç
---

## 👨‍💻 Geliştirici
**Oğuz Umut Meriç** - Software Engineering Student
[LinkedIn](https://linkedin.com/in/oguzzmeric) | [GitHub](https://github.com/oguzzmeric)
