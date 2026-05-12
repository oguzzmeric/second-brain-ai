import os
import time
from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool
from langchain_community.tools import DuckDuckGoSearchRun
from langchain_community.document_loaders import WebBaseLoader
from langchain_classic.agents import create_tool_calling_agent, AgentExecutor
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_openai import ChatOpenAI
from google.api_core.exceptions import ResourceExhausted

load_dotenv()


os.environ['USER_AGENT'] = "SecondBrain/1.0"


embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")


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
    collection_name = "second_brain_collection"
    
    if not os.path.exists(db_dir):
        return "Hata: Veritabanı bulunamadı. Lütfen önce döküman yükleyin."
    
    vectordb = Chroma(
        persist_directory=db_dir,
        embedding_function=embedding,
        collection_name=collection_name
    )
    
    # Top-K parametresini kota dostu olması için 5 yaptık
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

tools = [search_local_pdf, web_search_tool, read_web_page]

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
    
    Sadece tek bir kelime dön.
    """  
    try:
        decision = guard_llm.invoke(bekci_prompt).content.strip().upper()
        return decision
    except:
        return "AGENT" 

# --- ANA AJAN FONKSİYONU ---
def ask_brain_agent(user_input):
    prompt = ChatPromptTemplate.from_messages([
        ("system", """Sen üst düzey bir teknik analiz asistanısın.
        
        İZLEMEN GEREKEN 5 KESİN KURAL:
        1. ÖNCE DÖKÜMAN: Soru ne olursa olsun, HER ZAMAN önce 'search_local_pdf' kullan.
        2. İNTERNET YASAK BÖLGESİ: Eğer soru dökümanla/makaleyle ilgiliyse 'web_search_tool' kullanma.
        3. DÖKÜMANDA YOKSA: Sadece dökümanda bilgi kesinlikle yoksa internete bak.
        4. CEVAP KALİTESİ: Dökümandaki teknik terimleri ve sayıları asla değiştirme.
        5. farklı kaynakardaki biligiler arasında bağlantı kurarak cevap üret. Yani sadece dökümandaki bilgiyi vermekle kalma, farklı kaynaklardaki bilgileri birbirleriyle ilişkilendir ve çıkarım yap.,
        6. ÇAPRAZ İLİŞKİ: Farklı kaynaklardaki bilgileri birbirleriyle ilişkilendir ve çıkarım yap."""),
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
        max_iterations=5 # Kota sınırını korumak için iterasyon sayısını sınırladım
    )
    
    
    max_retries = 3
    for i in range(max_retries):
        try:
            return agent_executor.stream({"input": user_input})
        except ResourceExhausted:
            if i < max_retries - 1:
                print("Kota aşıldı, 1 dakika bekleniyor...")
                time.sleep(60) # Kota dolduğunda bekle
                continue
            raise