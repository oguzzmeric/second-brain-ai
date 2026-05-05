# 🧠 Second Brain AI (İkinci Beyin)

Second Brain AI, tamamen yerel (local) ortamda çalışan, gizlilik odaklı bir RAG (Retrieval-Augmented Generation) sistemidir. Teknik PDF dökümanlarınızı sisteme yükleyerek, belgeleriniz üzerinde Llama 3 altyapısıyla Türkçe dilinde akıllı ve bağlama dayalı sorgulamalar yapmanızı sağlar.

## 🚀 Özellikler

- **Tamamen Yerel ve Güvenli:** Verileriniz hiçbir bulut sunucusuna gönderilmez. Llama 3 modeli cihazınızda lokal olarak çalışır (Ollama).
- **Asimetrik Vektör Arama:** `intfloat/multilingual-e5-base` yerleştirme (embedding) modeli kullanılarak, sorgular asimetrik `query: ` prefix'i ile en yüksek doğrulukla eşleştirilir.
- **Dinamik Veritabanı Yönetimi:** UI üzerinden yeni döküman yüklendiğinde eski ChromaDB koleksiyonu otomatik olarak temizlenir ve Windows "WinError 32" dosya kilitlenme sorunları yaşanmaz.
- **Zorunlu Türkçe Çıktı:** Prompt mühendisliği sayesinde İngilizce ağırlıklı (English-dominant) LLM modelleri %100 Türkçe yanıt vermeye zorlanmıştır.
- **Kullanıcı Dostu Arayüz:** Streamlit ile geliştirilmiş modern, sürükle-bırak destekli web arayüzü.

## 🛠️ Teknolojiler ve Mimari

- **UI Framework:** Streamlit
- **LLM:** Meta Llama 3 (Ollama üzerinden)
- **Vektör Veritabanı:** ChromaDB
- **Embedding Modeli:** HuggingFace (`multilingual-e5-base`)
- **Orkestrasyon:** LangChain
- **Döküman İşleme:** PyMuPDF (`1000` chunk size, `300` overlap)

## 💻 Kurulum ve Çalıştırma

Sistemi çalıştırabilmek için aşağıdaki adımları izleyin:

### 1. Ön Koşullar
* Python 3.10 veya üzeri
* [Ollama](https://ollama.com/) cihazınızda kurulu olmalı.
* Ollama üzerinden Llama 3 modelini indirmiş olmalısınız:
  ```bash
  ollama run llama3
2. Repoyu Klonlayın
Bash
git clone [https://github.com/oguzzmeric/second-brain-ai.git](https://github.com/oguzzmeric/second-brain-ai.git)
cd second-brain-ai
3. Sanal Ortam Oluşturun ve Aktif Edin
Bash
python -m venv env

# Windows için:
.\env\Scripts\activate

# macOS/Linux için:
source env/bin/activate
4. Gerekli Kütüphaneleri Kurun
Bash
pip install -r requirements.txt
5. Uygulamayı Başlatın
Bash
streamlit run app2.py
📂 Proje Dosya Yapısı
app2.py: Streamlit arayüzünü ve ana akışı yöneten uygulama dosyası.

brain2.py: Vektör veritabanına bağlanıp, LLM ile sorguyu RAG zincirine sokan zeka katmanı.

ingest2.py: PDF dosyalarını okuyan, parçalayan (chunking) ve ChromaDB'ye kaydeden veri işleme motoru.

requirements.txt: Projenin bağımlılıkları.

👨‍💻 Geliştirici
Oğuz Umut Meriç

Yazılım Mühendisi & AI/IoT Geliştiricisi
