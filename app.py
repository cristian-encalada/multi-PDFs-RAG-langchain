import os
from dotenv import load_dotenv
import streamlit as st
from PyPDF2 import PdfReader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.prompts import ChatPromptTemplate
from langchain_community.vectorstores import FAISS
from langchain.tools.retriever import create_retriever_tool
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain.agents import AgentExecutor, create_tool_calling_agent


# load_dotenv()

# OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Load secrets from Streamlit
OPENAI_API_KEY = st.secrets["openai"]["api_key"]

embeddings = OpenAIEmbeddings(api_key=OPENAI_API_KEY)

def pdf_read(pdf_doc):
    text = ""
    for pdf in pdf_doc:
        pdf_reader = PdfReader(pdf)
        for page in pdf_reader.pages:
            text += page.extract_text()
    return text


def get_chunks(text):
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    chunks = text_splitter.split_text(text)
    return chunks


def vector_store(text_chunks):
    
    vector_store = FAISS.from_texts(text_chunks, embedding=embeddings)
    vector_store.save_local("faiss_db")


def get_conversational_chain(tools,ques):
    llm = ChatOpenAI(model_name="gpt-4o", temperature=0, api_key=OPENAI_API_KEY)
    prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """You are an assistant specialized in insurance policies. Your task is to provide accurate and detailed answers based on the context provided from multiple PDF documents. If the answer to a question is not available in the provided context, clearly state, 'The answer is not available in the context.' Do not attempt to provide an answer that is not supported by the given information.""",
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{input}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
)
    tool=[tools]
    agent = create_tool_calling_agent(llm, tool, prompt)

    agent_executor = AgentExecutor(agent=agent, tools=tool, verbose=True)
    response=agent_executor.invoke({"input": ques})
    print(response)
    st.write("Reply: ", response['output'])



def user_input(user_question):
    
    new_db = FAISS.load_local("faiss_db", embeddings,allow_dangerous_deserialization=True)
    
    retriever=new_db.as_retriever()
    retrieval_chain= create_retriever_tool(retriever,"pdf_extractor","This tool is to give answer to queries from the pdf")
    get_conversational_chain(retrieval_chain,user_question)


def main():
    st.set_page_config("Chat PDF")
    st.header("RAG based Chat with PDF")

    user_question = st.text_input("Ask a Question from the PDF Files")

    if user_question:
        user_input(user_question)

    with st.sidebar:
        st.title("Menu:")
        pdf_doc = st.file_uploader("Upload your PDF Files and Click on the Submit & Process Button", accept_multiple_files=True)
        if st.button("Submit & Process"):
            with st.spinner("Processing..."):
                raw_text = pdf_read(pdf_doc)
                text_chunks = get_chunks(raw_text)
                vector_store(text_chunks)
                st.success("Done")

if __name__ == "__main__":
    main()