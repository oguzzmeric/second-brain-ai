# 🧠 Second Brain AI V2: Agentic RAG Ecosystem

**Second Brain AI V2**, standart döküman sorgulama botlarının ötesine geçerek; **Otonom Ajan (Agentic)** mimarisi ve **Oturum Tabanlı Veri İzolasyonu** ile geliştirilmiş, kurumsal standartlarda bir dijital asistan sistemidir.

🚀 **Live Demo:** [second-brain-ai-i3.streamlit.app](https://second-brain-ai-i3.streamlit.app/)

---

## 🚀 v2 ile Gelen Yenilikler (Evolution)

* **Agentic Workflow:** LangChain "Tool Calling" altyapısı sayesinde sistem; soruyu analiz ederek teknik dökümanlarda (Internal PDF Search) mı yoksa internette (Web Search) mi araştırma yapacağına otonom olarak karar verir.
* **Multi-user Isolation (Zero-Footprint):** `tempfile` ve `Session State` entegrasyonu sayesinde her kullanıcının yüklediği dökümanlar tamamen izole edilir. Oturum kapandığında veriler otomatik temizlenir.
* **Hybrid Intelligence:** Karar mekanizması ve kompleks analizler için **GPT-4o-mini**, basit sohbet ve yönlendirme (Guard Route) görevleri için **GPT-3.5-Turbo** kullanılarak performans ve maliyet optimizasyonu sağlanmıştır.
* **Cross-Document Reasoning:** Geliştirilmiş $k = 20$ retrieval parametresi ile sistem, farklı dökümanlardaki bilgileri birbirleriyle ilişkilendirerek çıkarım yapar.
* **Cloud Native Deployment:** Streamlit Cloud üzerinde CI/CD süreçleriyle tamamen canlıya alınmış, erişilebilir bir yapıya kavuşmuştur.

---

## 🛠️ Teknik Yığın (Tech Stack)

| Katman | Teknoloji |
| :--- | :--- |
| **Zeka (LLM)** | OpenAI GPT-4o-mini & GPT-3.5-Turbo |
| **Orkestrasyon** | LangChain (Agentic Tool Calling & Chains) |
| **Vektör Veritabanı** | ChromaDB (In-memory & Session-based) |
| **Embedding** | HuggingFace `intfloat/multilingual-e5-base` |
| **Arayüz** | Streamlit (Custom Logic & Session Management) |

---

## 📐 Mimari Akış

1.  **Semantic Routing:** Kullanıcı girişi bir "Bekçi (Guard)" LLM tarafından analiz edilir (Complex vs Simple Chat).
2.  **Context Retrieval:** Teknik sorularda ajan, kullanıcıya özel oluşturulmuş `db_path` üzerinden vektör taraması yapar.
3.  **Web Augmentation:** Eğer dökümanlarda bilgi bulunamazsa, ajan internet arama araçlarını kullanarak güncel veriyi çeker.
4.  **Synthesis:** Toplanan tüm bağlam (Context) çapraz kontrol edilerek halüsinasyondan arındırılmış final yanıtı üretilir.

---

## 💻 Kurulum ve Çalıştırma

### 1. Çevresel Değişkenler
`.env` dosyanızı oluşturun ve anahtarınızı ekleyin:
`OPENAI_API_KEY=your_api_key_here`

### 2. Bağımlılıkları Yükleme
`pip install -r requirements.txt`

### 3. Uygulamayı Başlatma
`streamlit run app2.py`

---

## 📂 Proje Yapısı

- **app2.py**: Ana giriş noktası. UI yönetimi, oturum bazlı veri izolasyonu ve uygulama mantığı.
- **brain2.py**: Mantık katmanı. Otonom ajan (Agent), özel araçlar (Tools) ve hibrit yönlendirme sistemi.
- **ingest2.py**: Veri işleme motoru. PDF yükleme, metin parçalama (Chunking) ve vektör veritabanı inşası.

---

**Geliştirici:** [Oğuz Umut Meriç](https://www.linkedin.com/in/o%C4%9Fuz-meri%C3%A7-983a6b218/)  
**Vizyon:** "Bir döküman yığınından fazlası; otonom bir akıl."
