import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def run_ingestion():
    data_dir = "data/"
    db_dir = "chroma_db"
    collection_name = "second_brain_collection"

    # 1. Klasör Kontrolü
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return "Klasör oluşturuldu, lütfen içine PDF yükleyin."
    
    # 2. PDF Dosyalarını Listele
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    if not pdf_files:
        return "Hata: 'data/' klasöründe herhangi bir PDF dosyası bulunamadı."

    # 3. PDF İçeriklerini Yükle
    all_docs = []
    print(f"[SİSTEM] {len(pdf_files)} adet dosya okunuyor...")
    for pdf in pdf_files:
        try:
            loader = PyMuPDFLoader(os.path.join(data_dir, pdf))
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"[HATA] {pdf} okunurken sorun çıktı: {e}")

    #chunking 
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = text_splitter.split_documents(all_docs)
    
    if not chunks:
        return "Hata: PDF'lerden anlamlı bir metin çıkarılamadı."

    # 5. Embedding Model 
    embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    try:
        # 6. Eski Verileri Temizle
        # Windows kilitlenmeleri yaşadım(çözüm)
        if os.path.exists(db_dir):
            temp_db = Chroma(
                persist_directory=db_dir, 
                embedding_function=embedding,
                collection_name=collection_name
            )
            temp_db.delete_collection()
            print(f"[SİSTEM] '{collection_name}' isimli eski koleksiyon başarıyla silindi.")
        
        # 7. Yeni Veritabanı 
        print(f"[SİSTEM] {len(chunks)} yeni parça veritabanına yazılıyor...")
        Chroma.from_documents(
            documents=chunks,
            embedding=embedding,
            persist_directory=db_dir,
            collection_name=collection_name
        )
        
        return f"Başarılı! {len(chunks)} parça işlendi. Yeni makale sisteme tanımlandı."

    except Exception as e:
        return f"Veritabanı güncellenirken bir hata oluştu: {str(e)}"

if __name__ == "__main__":
    # Manuel test için
    print(run_ingestion())