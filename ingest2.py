import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

# Dışarıdan klasör yollarını alabilmesi için parametreleri ekledik
def run_ingestion(source_dir="data/", persist_dir="chroma_db"):
    collection_name = "second_brain_collection"

    # 1. Klasör Kontrolü
    if not os.path.exists(source_dir):
        os.makedirs(source_dir)
        return "Klasör oluşturuldu, lütfen içine PDF yükleyin."
    
    # 2. PDF Dosyalarını Listele
    pdf_files = [f for f in os.listdir(source_dir) if f.endswith(".pdf")]
    if not pdf_files:
        return f"Hata: '{source_dir}' klasöründe herhangi bir PDF dosyası bulunamadı."

    # 3. PDF İçeriklerini Yükle
    all_docs = []
    print(f"[SİSTEM] {len(pdf_files)} adet dosya okunuyor...")
    for pdf in pdf_files:
        try:
            # Dinamik source_dir kullanıyoruz
            loader = PyMuPDFLoader(os.path.join(source_dir, pdf))
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"[HATA] {pdf} okunurken sorun çıktı: {e}")

    # 4. Chunking 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = text_splitter.split_documents(all_docs)
    
    if not chunks:
        return "Hata: PDF'lerden anlamlı bir metin çıkarılamadı."

    # 5. Embedding Model 
    embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    try:
        # 6. Eski Verileri Temizle (Kullanıcıya özel klasörde)
        if os.path.exists(persist_dir):
            temp_db = Chroma(
                persist_directory=persist_dir, 
                embedding_function=embedding,
                collection_name=collection_name
            )
            temp_db.delete_collection()
            print(f"[SİSTEM] '{collection_name}' isimli eski koleksiyon temizlendi.")
        
        # 7. Yeni Veritabanı (Kullanıcıya özel persist_dir içine)
        print(f"[SİSTEM] {len(chunks)} yeni parça veritabanına yazılıyor...")
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=persist_dir,
            collection_name=collection_name
        )
        
        return f"Başarılı! {len(chunks)} parça işlendi. Sistem size özel verilerle güncellendi."

    except Exception as e:
        return f"Veritabanı güncellenirken bir hata oluştu: {str(e)}"

if __name__ == "__main__":
    # Manuel test hala çalışır
    print(run_ingestion())