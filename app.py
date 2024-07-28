import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from langchain.chains import ConversationalRetrievalChain
from langchain_chroma import Chroma
from langchain_groq import ChatGroq
from langchain.prompts import PromptTemplate
from langchain.schema import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.embeddings import OpenAIEmbeddings
from collections import deque
import os
import datasets
import asyncio
from functools import partial
import traceback
import time
HUGGINGFACE_API_TOKEN= 'API Key Here'
GROQ_API_KEY= 'API Key Here'
OPENAI_API_KEY= 'API Key Here'
@st.cache_resource
def load_data_and_retriever():
    data = datasets.load_dataset("adamo1139/basic_economics_questions_ts_test_4")
    data = data['train']['text']
    formatted_data = [{"page_content": text.replace("<s>", "").replace("</s>", "")} for text in data]
    documents = [Document(page_content=item['page_content'], metadata={}) for item in formatted_data]

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    splits = text_splitter.split_documents(documents)
    
    embedding_model = OpenAIEmbeddings()
    vector_embed = Chroma.from_documents(documents=splits, embedding=embedding_model)
    retriever = vector_embed.as_retriever(search_kwargs={"k": 3})

    return retriever

def initialize_retrieval_chain(model, retriever):
    prompt_template = """
                You are an expert economics chatbot. Your task is to provide information-rich, yet concise answers to questions based on the given context. Avoid giving overly detailed answers unless explicitly asked for more detail. If a question falls outside the domain of economics, respond that you don't know the answer. Additionally, do not mention Thomas Sowell or directly quote him in your responses.

    {context}

    Human: {question}
    AI:"""

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context", "question"]
    )

    chain = ConversationalRetrievalChain.from_llm(
        llm=model,
        retriever=retriever,
        combine_docs_chain_kwargs={"prompt": PROMPT},
        return_source_documents=False,
        max_tokens_limit=4000
    )

    return chain

async def async_get_response(query, chat_history, chain):
    loop = asyncio.get_event_loop()
    response = await loop.run_in_executor(None, partial(chain, {"question": query, "chat_history": chat_history}))
    return response['answer']

def main():
    st.set_page_config(page_title="EconChat", page_icon="💹")
    st.title("EconChat")
    st.write("Welcome to EconChat! Think Like An Economist")

    retriever = load_data_and_retriever()
    model = ChatGroq(model="llama3-8b-8192")
    chain = initialize_retrieval_chain(model, retriever)

    if 'chat_history' not in st.session_state:
        st.session_state.chat_history = deque(maxlen=10)  # Limit history to last 10 exchanges

    for message in st.session_state.chat_history:
        if isinstance(message, HumanMessage):
            st.markdown(f"**You:** {message.content}")
        else:
            st.markdown(f"**EconChat:** {message.content}")

    user_input = st.text_input("Your Question:", key="user_input")

    if st.button("Send"):
        if user_input:
            with st.spinner('EconChat is thinking...'):
                chat_history = list(st.session_state.chat_history)

                try:
                    answer = chain({"question": user_input, "chat_history": chat_history})['answer']

                    st.session_state.chat_history.append(HumanMessage(content=user_input))
                    st.session_state.chat_history.append(AIMessage(content=answer))

                    st.markdown(f"**You:** {user_input}")
                    st.markdown(f"**EconChat:** {answer}")

                    # Add a delay to avoid rate limiting
                    time.sleep(1)

                except Exception as e:
                    st.error(f"An error occurred: {str(e)}")
                    st.error(traceback.format_exc())

            st.rerun()

if __name__ == "__main__":
    main()