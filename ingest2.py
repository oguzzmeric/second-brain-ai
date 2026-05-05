import os
import shutil
from langchain_community.document_loaders import PyMuPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_chroma import Chroma

def run_ingestion():
    # KLASÖR KONTROLÜ BURADA (Fonksiyon çalışınca devreye girer)
    data_dir = "data/"
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)
        return "Klasör oluşturuldu, lütfen PDF yükleyin."
    
    # 9. SATIR BURAYA TAŞINDI: Artık patlamaz
    pdf_files = [f for f in os.listdir(data_dir) if f.endswith(".pdf")]
    
    if not pdf_files:
        return "Hata: 'data/' klasöründe PDF bulunamadı."

    all_docs = []
    for pdf in pdf_files:
        try:
            loader = PyMuPDFLoader(os.path.join(data_dir, pdf))
            all_docs.extend(loader.load())
        except Exception as e:
            print(f"Hata: {e}")

    # Teknik döküman ayarları (1000/300)
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=300)
    chunks = text_splitter.split_documents(all_docs)
    
    # Asimetrik Arama İçin E5-Base
    embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

    if os.path.exists("chroma_db"):
        try:
            eski_db = Chroma(persist_directory="chroma_db", embedding_function=embedding)
            eski_db.delete_collection()
        except Exception:
            pass

    Chroma.from_documents(
        documents=chunks,
        embedding=embedding,
        persist_directory="chroma_db"
    )
    return f"Başarılı! {len(chunks)} parça işlendi."