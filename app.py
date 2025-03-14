# requirements.txt
# streamlit
# beautifulsoup4
# langchain
# openai
# python-dotenv
# faiss-cpu
# arabic-reshaper
# python-bidi
# googletrans==4.0.0-rc1

import os
import streamlit as st
from langchain.document_loaders import WebBaseLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema.runnable import RunnablePassthrough
from arabic_reshaper import reshape
from bidi.algorithm import get_display
import base64
import requests

# Configure page
st.set_page_config(
    page_title="Arabic AI Agent",
    page_icon="🤖",
    layout="centered",
    initial_sidebar_state="expanded"
)

# Arabic display functions
def arabic_text(text):
    reshaped = reshape(text)
    bidi_text = get_display(reshaped)
    return f'<p style="text-align: right; direction: rtl; font-family: Arial;">{bidi_text}</p>'

# URL processing functions
def process_url(url):
    loader = WebBaseLoader(url)
    docs = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200
    )
    splits = text_splitter.split_documents(docs)
    
    embeddings = OpenAIEmbeddings()
    vectorstore = FAISS.from_documents(splits, embeddings)
    
    return vectorstore.as_retriever()

# AI response generation
def generate_response(query, retriever):
    template = """Answer in Arabic using the following context:
    {context}
    
    Question: {question}
    """
    
    prompt = ChatPromptTemplate.from_template(template)
    llm = ChatOpenAI(model="gpt-4-1106-preview", temperature=0)
    
    chain = (
        {"context": retriever, "question": RunnablePassthrough()}
        | prompt
        | llm
    )
    
    return chain.invoke(query)

# Sharing functions
def get_share_link():
    params = st.experimental_get_query_params()
    return f"https://share.streamlit.io/your-app-url?query={params}"

# Main app
def main():
    st.title("🤖 Arabic AI Agent with URL Processing")
    
    with st.sidebar:
        st.header("Configuration")
        api_key = st.text_input("Enter OpenAI API Key:", type="password")
        if api_key:
            os.environ["OPENAI_API_KEY"] = api_key
    
    tab1, tab2, tab3 = st.tabs(["Process URL", "Chat Interface", "Share Agent"])
    
    with tab1:
        st.subheader("Train Agent on Website Content")
        url = st.text_input("Enter website URL:")
        
        if url:
            with st.spinner("Processing Arabic content..."):
                try:
                    retriever = process_url(url)
                    st.session_state.retriever = retriever
                    st.success("Agent trained successfully on website content!")
                except Exception as e:
                    st.error(f"Error processing URL: {str(e)}")
    
    with tab2:
        st.subheader("Chat with Arabic Agent")
        
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(arabic_text(message["content"]), unsafe_allow_html=True)
        
        if prompt := st.chat_input("Ask in Arabic or English"):
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(arabic_text(prompt), unsafe_allow_html=True)
            
            with st.spinner("Generating Arabic response..."):
                try:
                    response = generate_response(prompt, st.session_state.get("retriever", None))
                    st.session_state.messages.append({"role": "assistant", "content": response.content})
                    with st.chat_message("assistant"):
                        st.markdown(arabic_text(response.content), unsafe_allow_html=True)
                except Exception as e:
                    st.error(f"Error generating response: {str(e)}")
    
    with tab3:
        st.subheader("Share Your Agent")
        st.markdown("""
        ### Share Options
        1. **Direct Link** - Share this URL with collaborators:
        """)
        st.code(get_share_link(), language="text")
        
        st.markdown("""
        2. **Embed in Website** - Add this iframe to your HTML:
        """)
        embed_code = f"""
        <iframe src="{get_share_link()}"
            width="100%"
            height="600"
            frameborder="0"
            allow="microphone">
        </iframe>
        """
        st.code(embed_code, language="html")
        
        st.markdown("""
        3. **API Access** - Integrate with our REST API:
        """)
        st.code("""
        POST /api/v1/agent
        Headers: {"Authorization": "Bearer YOUR_API_KEY"}
        Body: {"query": "Your question in Arabic"}
        """, language="json")

if __name__ == "__main__":
    main()
