from langchain.llms import GooglePalm
from dotenv import load_dotenv
import os
from langchain.document_loaders.csv_loader import CSVLoader
from langchain.embeddings import HuggingFaceInstructEmbeddings
from langchain.vectorstores import FAISS
from langchain.prompts import PromptTemplate
from langchain.chains import RetrievalQA



load_dotenv()
api_key = os.getenv('GEMINI_API_KEY')

llm = GooglePalm(google_api_key=api_key,temperature=0)







instructor_embeddings = HuggingFaceInstructEmbeddings()

#save vectordb to disk in a file as there is no point to run it again and again for each query

vectordb_file_path = "faiss_index"

def create_vector_db():
    loader = CSVLoader(file_path='faqs.csv',source_column='prompt',encoding='cp1252')
    data = loader.load()

    vectordb = FAISS.from_documents(
        documents=data,
        embedding=instructor_embeddings
        )
    
    vectordb.save_local(vectordb_file_path)

def get_qa_chain():
    #Lod vector db from local folder
    vectordb = FAISS.load_local(vectordb_file_path,instructor_embeddings)

    #Create a retriever for querying the vector db
    retriever = vectordb.as_retriever(score_threshold = 0.8)

    prompt_template = """
    Given the following context and a question, generate an answer based on this context.
    In the answer try to provide as much text as possible from "response" section in the source document provided.
    If the answer is not found in the context, kindly state "Sorry,I don't know." Don't try to make up an answer on your own

    CONTEXT: {context}

    QUESTION: {question}

    """

    PROMPT = PromptTemplate(
        template=prompt_template, input_variables=["context","question"]
    )

    chain = RetrievalQA.from_chain_type(
        llm = llm,
        chain_type="stuff",
        retriever=retriever,
        return_source_documents=True,
        chain_type_kwargs={"prompt":PROMPT}
    )

    return chain
    

if __name__ == "__main__":
    # create_vector_db()
    chain = get_qa_chain()
