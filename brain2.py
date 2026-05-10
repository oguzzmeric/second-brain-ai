import os
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import ChatOllama
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor

# Web scraping için user agent tanımlaması şart
os.environ['USER_AGENT'] = "SecondBrain/1.0"

# 1. Embeddings ve LLM Kurulumu
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")

llm = ChatOllama(
    model="llama3.1", 
    temperature=0.1 # Stabilite için düşük tutuyoruz.
)

# 2. Araçların (Tools) Tanımlanması

# İnternet Arama Aracı
web_search_tool = DuckDuckGoSearchRun(
    name='web_search_tool',
    description='Hava durumu, güncel haberler veya PDF dökümanlarında BULUNMAYAN genel bilgiler için bu aracı kullan.'
)

@tool
def read_web_page(url: str) -> str:
    """İnternet aramasından çıkan bir URL'in içeriğini okumak için kullanılır."""
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return data[0].page_content[:4000]
    except Exception as e:
        return f"Web sayfası okunurken hata oluştu: {str(e)}"

@tool
def search_local_pdf(query: str) -> str:
    """Yalnızca yüklü olan teknik dökümanlarda (PDF) detaylı arama yapar."""
    db_dir = "chroma_db"
    # KRİTİK: ingest2.py ile aynı koleksiyon ismi olmalı!
    collection_name = "second_brain_collection"
    
    if not os.path.exists(db_dir):
        return "Hata: Veritabanı bulunamadı. Lütfen önce döküman yükleyin."
    
    # Veritabanına bağlanıyoruz
    vectordb = Chroma(
        persist_directory=db_dir, 
        embedding_function=embedding,
        collection_name=collection_name
    )
    
    # k: 10 -> Daha geniş tarama yaparak detayları kaçırmamasını sağlıyoruz
    retriever = vectordb.as_retriever(search_kwargs={"k": 10})
    docs = retriever.invoke(query)
    
    # Terminal Logları (Hata ayıklama için)
    print(f"\n[DEBUG] Arama Sorgusu: {query}")
    print(f"[DEBUG] Bulunan Parça Sayısı: {len(docs)}")
    
    if len(docs) > 0:
        print(f"[DEBUG] İlk Parçadan Örnek: {docs[0].page_content[:150]}...")
        return "\n\n".join(doc.page_content for doc in docs)
    else:
        return "Dökümanda bu konuyla ilgili hiçbir bilgi bulunamadı."

# Ajanın kullanabileceği araç çantası
# Not: web_search_tool'u buraya tekrar ekledim ama prompt ile dizginledik.
tools = [search_local_pdf, web_search_tool, read_web_page]

# 3. Ajan Fonksiyonu
def ask_brain_agent(user_input):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Sen üst düzey bir teknik analiz asistanısın. 
        
        İZLEMEN GEREKEN KESİN KURALLAR:
        1. ÖNCE DÖKÜMAN: Soru ne olursa olsun (özet veya detay), HER ZAMAN önce 'search_local_pdf' aracını kullan.
        2. İNTERNET YASAK BÖLGESİ: Eğer soru dökümanla/makaleyle/şartnameyle ilgiliyse 'web_search_tool' kullanma.
        3. DÖKÜMANDA YOKSA: Sadece dökümanda bilgi kesinlikle yoksa veya soru tamamen döküman dışıysa internete bak.
        4. CEVAP KALİTESİ: Dökümandaki teknik terimleri, sayıları ve görev tanımlarını asla değiştirme.
        5. 'Yeni Teknolojiler ve Ekonomi' gibi alakasız internet sonuçlarını döküman bilgisiymiş gibi sunma."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    llm_with_tools = llm.bind_tools(tools)
    agent = create_tool_calling_agent(llm_with_tools, tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent, 
        tools=tools, 
        verbose=True, 
        handle_parsing_errors=True,
        max_iterations=5
    )

    return agent_executor.stream({"input": user_input})