import streamlit as st
import os
from langchain_groq import ChatGroq 
from langchain_core.messages import HumanMessage, AIMessage
from langchain_community.document_loaders import WebBaseLoader 
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import FastEmbedEmbeddings

st.set_page_config(
    page_title="TasHus Live Website Assistant", 
    page_icon="🚘", 
    layout="centered"
)

if "GROQ_API_KEY" in st.secrets:
    GROQ_API_KEY = st.secrets["GROQ_API_KEY"]
else:
    GROQ_API_KEY = os.environ.get("GROQ_API_KEY", "")

st.markdown("""
<style>
        .stApp { background-color: #f8f9fa; }
        .hero-banner {
            background: linear-gradient(135deg, #a435f0 0%, #5022c3 100%);
            padding: 2.5rem 2rem;
            border-radius: 16px;
            color: white;
            text-align: center;
            border-bottom: 5px solid #c0c4fc;
            box-shadow: 0 10px 25px rgba(164, 53, 240, 0.3);
            margin-bottom: 2rem;
        }
        .hero-banner h1 { color: white !important; font-size: 2.5rem !important; font-weight: 800 !important; }
        .hero-banner p { color: #c0c4fc !important; font-size: 1.1rem !important; font-weight: 500 !important; }
        .stChatInput { border-radius: 30px !important; box-shadow: 0 4px 15px rgba(164, 53, 240, 0.1) !important; }
    </style>
""", unsafe_allow_html=True)

st.markdown("""
    <div class="hero-banner">
        <h1>TasHus Assistant</h1>
        <p>Your smart workspace conversational tool powered by Siara Solutions</p>
    </div>
""", unsafe_allow_html=True)


# ১ সেকেন্ড ক্যাশ দেওয়ার ফলে প্রতিবার লাইভ সাইটের নতুন ডাটা রিয়েল-টাইম স্ক্র্যাপ হবে
@st.cache_resource(ttl=1) 
def build_web_knowledge_base():
    # আপনার ওয়েবসাইটের সমস্ত গুরুত্বপূর্ণ পেজের রিয়েল-টাইম ডাটা স্ক্র্যাপ করার জন্য লিংকসমূহ
    urls = [
        "https://tashus.com.au",
        "https://tashus.com.au",
        "https://tashus.com.au",
        "https://tashus.com.au",
        "https://tashus.com.au"
    ]
    
    all_docs = []
    try:
        loader = WebBaseLoader(urls)
        all_docs = loader.load()
    except Exception as e:
        st.error(f"Website Scraping Error: Could not read links! {str(e)}")
        st.stop()
  
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=120)
    final_chunks = text_splitter.split_documents(all_docs)
    
    embeddings = FastEmbedEmbeddings(model_name="BAAI/bge-small-en-v1.5")
    
    vector_db = FAISS.from_documents(final_chunks, embeddings)
    return vector_db.as_retriever(search_kwargs={"k": 4})


with st.status("Fetching and indexing live data from TasHus website...", expanded=False) as status:
    retriever = build_web_knowledge_base()
    status.update(label="All Live Website Data Successfully Cached!", state="complete", expanded=False)


if not GROQ_API_KEY:
    st.error("API Key Error: Please set GROQ_API_KEY in Streamlit Secrets or Environment Variables.")
    st.stop()

# temperature=0.0 করা হয়েছে নিখুঁত উত্তরের জন্য এবং হ্যালুসিনেশন রোধে
llm = ChatGroq(
    model="llama-3.3-70b-versatile", 
    groq_api_key=GROQ_API_KEY,
    temperature=0.0 
)

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

for message in st.session_state.chat_history:
    if isinstance(message, HumanMessage):
        with st.chat_message("user", avatar="👤"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant", avatar="🚘"):
            st.markdown(message.content)


if user_query := st.chat_input("Ask a question about TasHus services..."):
    
    with st.chat_message("user", avatar="👤"):
        st.markdown(user_query)
        
    matching_pdf_data = retriever.invoke(user_query)
    extracted_context = "\n\n".join([doc.page_content for doc in matching_pdf_data])
    
    agent_prompt = ChatPromptTemplate.from_messages([
        ("system", (
            "You are an expert customer service concierge for TasHus Car Rental in Australia.\n"
            "Formulate friendly, comprehensive, and highly precise answers using exclusively the document context below.\n\n"
            
            "CRITICAL ORDER OF RESPONSE FOR VEHICLES:\n"
            "When a user asks for details or specifications of a car (like Toyota Hiace, Hyundai Accent, or Mitsubishi Pajero):\n"
            "1. FIRST, provide a detailed description including its Type, Key Features, and Specifications based ONLY on the facts in the context.\n"
            "2. SECOND, at the very end of your response, politely suggest the official page link in Markdown format: [Vehicle Details](URL).\n"
            "3. NEVER just give the link alone. Always provide the full descriptive details first as requested by the user.\n\n"

            "GENERAL RULES:\n"
            "- You must answer questions using EXCLUSIVELY the facts explicitly stated in the PROVIDED DOCUMENT CONTEXT below. Do not use external knowledge.\n"
            "- If the user asks about discounts, rates, or information NOT explicitly written in the context below, you are FORBIDDEN from mentioning or guessing any numbers, percentages, or terms.\n"
            "- IF THE INFORMATION IS COMPLETELY MISSING, RESPOND EXACTLY WITH:\n"
            "'I apologize, but I cannot find that information in our current documentation files.'\n\n"
            
            "OFFICIAL TASHUS WEBSITE LINKS:\n"
            "- Main Website: https://tashus.com.au\n"
            "- Privacy Policy Page: https://tashus.com.au\n"
            "- Terms & Conditions: https://tashus.com.au\n"
            "- User Verification/Account Registration: https://tashus.com.au\n"
            "- Contact Support: https://tashus.com.au\n"
            "- Find Car/Vehicle Search: https://tashus.com.au\n"
            "- Toyota Hiace: https://tashus.com.au\n"
            "- 2011 Hyundai Accent Hatchback: https://tashus.com.au\n" 
            "- 2015 Mitsubishi Pajero: https://tashus.com.au\n\n"
            
            "If the text query is about content rules but does NOT match any known website link above, "
            "and it isn't explicitly detailed in the context, say: "
            "'I apologize, but I cannot find that information in our current documentation files.'\n\n"
            f"PROVIDED DOCUMENT CONTEXT:\n{extracted_context}"
        )),
        MessagesPlaceholder(variable_name="chat_history"),
        ("human", "{input}")
    ])

    processing_chain = agent_prompt | llm

    with st.chat_message("assistant", avatar="🚘"):
        with st.spinner("Analyzing web data archives..."):
            ai_response = processing_chain.invoke({
                "input": user_query,
                "chat_history": st.session_state.chat_history
            })
            st.markdown(ai_response.content)
        
    st.session_state.chat_history.append(HumanMessage(content=user_query))
    st.session_state.chat_history.append(AIMessage(content=ai_response.content))

