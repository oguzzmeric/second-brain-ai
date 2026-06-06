import os
import time
import streamlit as st # Streamlit eklendi
from functools import partial
from langchain_chroma import Chroma
# HuggingFace yerine OpenAI Embeddings import ediyoruz
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
from langchain_core.prompts import MessagesPlaceholder

load_dotenv()

os.environ['USER_AGENT'] = "SecondBrain/1.0"

# --- BELLEK OPTİMİZASYONU (CACHE) ---
@st.cache_resource
def get_embedding_model():
    """Embedding modelini belleğe bir kez yükler ve orada tutar."""
    # Eski 768 boyutlu HF modeli yerine 1536 boyutlu OpenAI modeline geçtik
    return OpenAIEmbeddings(model="text-embedding-3-small")

# Modeli her seferinde baştan oluşturmak yerine cache'den çekiyoruz
embedding = get_embedding_model()
# ------------------------------------

llm = ChatOpenAI(
    model="gpt-4o-mini",
    temperature=0
)

web_search_tool = DuckDuckGoSearchRun(
    name='web_search_tool',
    description='Hava durumu, güncel haberler veya dökümanlarda BULUNMAYAN genel bilgiler için kullan.'
)

@tool
def read_web_page(url: str) -> str:
    """SADECE ve SADECE kullanıcının sağladığı geçerli bir 'http...' internet linki (URL) varsa bu aracı kullan. Elinde URL yoksa ASLA kullanma."""
    try:
        loader = WebBaseLoader(url)
        data = loader.load()
        return data[0].page_content[:4000]
    except Exception as e:
        return f"Web sayfası okunurken hata oluştu: {str(e)}"

def search_local_pdf(query: str, db_path: str = "chroma_db") -> str:
    """Yalnızca yüklü olan teknik dökümanlarda (PDF) detaylı arama yapar."""
    collection_name = "second_brain_collection"
    
    if not os.path.exists(db_path):
        return "Hata: Veritabanı bulunamadı. Lütfen önce döküman yükleyin."
    
    vectordb = Chroma(
        persist_directory=db_path,
        embedding_function=embedding,
        collection_name=collection_name
    )
    
    retriever = vectordb.as_retriever(search_kwargs={"k": 20})
    docs = retriever.invoke(query)
    
    if len(docs) > 0:
        formatted_docs = []
        for doc in docs:
            content_path = doc.metadata.get("source", "bilinmeyen kaynak")
            doc_name = os.path.basename(content_path)
            page = doc.metadata.get("page", "bilinmeyen sayfa")
            text = doc.page_content
            formatted_docs.append(f"[kaynak]:{doc_name} [sayfa]:{page}\n{text}\n")
        return "\n---\n".join(formatted_docs)
    else:
        return "Dökümanda bu konuyla ilgili bilgi bulunamadı."

def get_document_names(db_path: str = "chroma_db") -> str:
    """Sisteme yüklenen dökümanları, isimlerini, sayfa sayılarını gibi bilgileri verir."""
    collection_name = "second_brain_collection"
    vector_db = Chroma(
        persist_directory=db_path,
        embedding_function=embedding,
        collection_name=collection_name
    )


    db_data = vector_db.get(include=["metadatas", "documents"])
    metadatas = db_data.get("metadatas", [])
    if not metadatas:
        return "Sistemde yüklü döküman bulunamadı."
    
    unique_docs = set()
    for meta in metadatas:
        source = meta.get("source")
        if source:
            unique_docs.add(os.path.basename(source))
    
    return "sistemde yüklü dökümanlar:\n" + "\n".join(unique_docs)
    

# --- HİBRİT YÖNLENDİRİCİ (GUARD ROUTE) ---
def guard_route(user_input):
    """Sorumluluk Reddi: Soruyu analiz eder ve rotayı belirler."""
    guard_llm = ChatOpenAI(
        model="gpt-3.5-turbo",
        temperature=0
    )

    bekci_prompt = f"""
    Sen bir yönlendirme asistanısın. Kullanıcının sorusunu analiz et.
    KULLANICI SORUSU: "{user_input}"
    
    KURALLAR:
    1. Eğer soru kompleks bir soru ise "AGENT" dön.
    2. Eğer soru selamlaşma (merhaba, nasılsın), sistemin ne olduğu veya genel sohbet ise SADECE "SIMPLE" dön.
    
    Sadece tek Kelime dön.
    """  
    try:
        decision = guard_llm.invoke(bekci_prompt).content.strip().upper()
        return decision
    except:
        return "AGENT" 

# --- ANA AJAN FONKSİYONU ---
def ask_brain_agent(user_input, db_path="chroma_db"):
    # NOT: Fonksiyon artık db_path parametresini kabul ediyor.
    
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Sen, kendisine özel bir yerel vektör veritabanı (ChromaDB) bağlanmış üst düzey bir teknik analiz asistanısın.

        KİMLİĞİN VE FARKINDALIĞIN (SELF-AWARENESS):
        - Senin kalbin ve ana bilgi kaynağın, kullanıcının sisteme yüklediği PDF'lerden oluşan bu yerel veritabanıdır.
        - Kullanıcı "döküman", "pdf", "veritabanı", "dosyalar" veya "sistem" dediğinde, DOĞRUDAN sana bağlı olan bu yerel veritabanından bahsedildiğini bil. Asla "Hangi döküman?" diye sorma.
        - Veritabanının içinde tam olarak hangi metinlerin olduğunu ezbere bilemezsin. İçeriyi görmek için HER ZAMAN alet çantandaki araçları (tools) kullanmak zorundasın.

        İZLEMEN GEREKEN 6 KESİN KURAL:
        1. ÖNCE KENDİ HAFIZAN: Kullanıcının sorusu ne olursa olsun, ilk işin 'search_local_pdf' aracını çalıştırıp kendi veritabanında arama yapmaktır.
        2. DÖKÜMAN LİSTESİ: Eğer kullanıcı "Hangi dökümanlar var?", "Neler yüklü?" gibi sistemin içeriğini sorarsa, ASLA uydurma. Hemen 'list_documents_tool' aracını kullan ve sadece oradan dönen gerçek listeyi ver.
        3. İNTERNET YASAK BÖLGESİ: Soru spesifik olarak "yüklü dökümanlarla" ilgiliyse internet (web_search_tool) KESİNLİKLE yasaktır. Dökümanda yoksa "Dökümanda bulamadım" de ve konuyu kapat.
        4. İNTERNETE ÇIKIŞ İZNİ: SADECE kullanıcı genel bir soru soruyorsa ve bu bilgi dökümanlarda kesinlikle yoksa internete (web_search_tool) bak.
        5. VERİ SADAKATİ: Veritabanından çektiğin metinlerdeki sayıları, formülleri ve teknik terimleri asla değiştirme, olduğu gibi aktar.
        6. VARLIK (ENTITY) KORUMASI: Farklı dökümanlardan bilgi çekerken kişileri, kurumları veya projeleri ASLA birbirine karıştırma. Eğer kullanıcı belirli bir kişinin (örn: Oğuz'un) eğitimini soruyorsa ve dökümanda o kişinin eğitimi yoksa, sistemdeki BAŞKA birinin eğitim bilgisini (örn: ODTÜ 1993) ona atfetme.
        7. BİLMİYORUM DEME ERDEMİ: Bir soruya ait net bir eşleşme bulamazsan çıkarım yapmaya veya boşluk doldurmaya zorlama. Direkt olarak "Dökümanlarda bu kişinin/konunun özelinde spesifik bir bilgi bulamadım" de."""),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ])

    # Kritik Kısım: search_local_pdf aracına db_path'i önceden enjekte ediyoruz.
    # Bu sayede Agent, arama yaparken otomatik olarak doğru klasöre bakacak.
    
    @tool
    def configured_search_tool(query: str) -> str:
        """Yalnızca yüklü olan teknik dökümanlarda (PDF) detaylı arama yapar."""
        # Dışarıdaki db_path'i kullanacak
        return search_local_pdf(query, db_path=db_path)
    
    @tool
    def list_documents_tool(query: str = "liste") -> str:
        """Sistemde HANGİ dökümanların, dosyaların veya PDF'lerin yüklü olduğunu, bunların isimlerini sorarsa KESİNLİKLE bu aracı kullan."""
        return get_document_names(db_path=db_path)

    current_tools = [configured_search_tool, list_documents_tool, web_search_tool, read_web_page]

    llm_with_tools = llm.bind_tools(current_tools)
    agent = create_tool_calling_agent(llm_with_tools, current_tools, prompt)
    
    agent_executor = AgentExecutor(
        agent=agent,
        tools=current_tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=5
    )
    
    max_retries = 3
    for i in range(max_retries):
        try:
            return agent_executor.stream({"input": user_input})
        except ResourceExhausted:
            if i < max_retries - 1:
                print("Kota aşıldı, 1 dakika bekleniyor...")
                time.sleep(60)
                continue
            raise