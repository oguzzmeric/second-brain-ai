from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

# embeddings and llm setup
embedding = HuggingFaceEmbeddings(model_name="intfloat/multilingual-e5-base")
llm = OllamaLLM(model="llama3")


def ask_brain(query):
    # connect to chroma db
    vectordb = Chroma(persist_directory="chroma_db", embedding_function=embedding)
    retriever = vectordb.as_retriever(search_kwargs={"k": 6}) 
    
    system_prompt = (
        "Sen uzman bir yardımcı asistan ve teknik döküman analizcisisin. "
        "Sana sağlanan bağlamı kullanarak soruyu yanıtla.\n\n"
        "1. Sadece bağlamdaki verileri kullan.\n"
        "2. Akıcı bir Türkçe ile cevap ver.\n\n"
        "3.sadece %100 Türkçe cevap ver teknik terimler hariç"
        "Bağlam:\n{context}"
    )

    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        ("human", "{input}\n\n(Lütfen cevabını tamamen Türkçe dilde yaz.)"),
    ])

    def format_docs(docs):
        return "\n\n".join(doc.page_content for doc in docs)

    chain = (
        {"context": retriever | format_docs, "input": RunnablePassthrough()}
        | prompt | llm | StrOutputParser()
    )
    
    #E5 prefix 
    return chain.invoke(f"query: {query}")