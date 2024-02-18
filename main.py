import streamlit as st
from langchain_helper import create_vector_db,get_qa_chain

st.set_page_config(
    page_title="EdTech QA Bot",
    page_icon="🧑‍💻",
    layout="wide"
)

st.title("EdTech🧑‍💻 QA Bot🤖")

#create vector db for admin
# btn = st.button("Create Knowledge Base")

question = st.text_input("Question: ")

if question:
    chain = get_qa_chain()
    response = chain(question)
    print(response)

    st.header("Answer: ")
    st.write(response["result"])
