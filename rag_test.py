from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain.text_splitter import RecursiveCharacterTextSplitter
import os
from langchain_community.document_loaders import PyPDFLoader
import streamlit as st

os.environ["OPENAI_API_KEY"] = "sk-proj-8FqpllUlcdCvSxqgkV1Z0pIuJM-0i" \
                               "-SPUnPSPfP1LCEOY2fAbfwFYxOJ4JPb9uokAQmQj82lPtT3BlbkFJeCGgoxvPlSXu_nd2EGBJSG" \
                               "-FiQ5MHd3XDt2cbmtlkpj5_z_aCRVMl8-dkB4i4QF4I379II2HwA"
openai_api_key = os.getenv('OPENAI_API_KEY')
doc_path = "source_data"
chroma_path = "src/contracts_chroma"
pdf_folder_path = "source_data"
documents = []
for file in os.listdir(pdf_folder_path):
    if file.endswith('.pdf'):
        pdf_path = os.path.join(pdf_folder_path, file)
        loader = PyPDFLoader(pdf_path)
        documents.extend(loader.load())

# split the doc into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
chunks = text_splitter.split_documents(documents)

os.environ["OPENAI_API_KEY"] = openai_api_key

# get OpenAI Embedding model
embeddings = OpenAIEmbeddings(openai_api_key=os.environ["OPENAI_API_KEY"])

# embed the chunks as vectors and load them into the database
db_chroma = Chroma.from_documents(chunks, embeddings, persist_directory=chroma_path)


# this is an example of a user question (query)
def generate_response(input_text):
    # retrieve context - top 5 most relevant (closests) chunks to the query vector
    # (by default Langchain is using cosine distance metric)
    docs_chroma = db_chroma.similarity_search_with_score(input_text, k=5)

    # generate an answer based on given user query and retrieved context information
    context_text = "\n\n".join([doc.page_content for doc, _score in docs_chroma])

    # you can use a prompt template

    PROMPT_TEMPLATE = """
    Answer the question based only on the following context:
    {context}
    Answer the question based on the above context: {question}.
    Provide a detailed answer.
    Don’t justify your answers.
    If the question is in English, give the answers in english, if they are in Finnihs, answwers in Finnish
    Don’t give information not mentioned in the CONTEXT INFORMATION.
    Do not say "according to the context" or "mentioned in the context" or similar.
    """

    # load retrieved context and user query in the prompt template
    prompt_template = ChatPromptTemplate.from_template(PROMPT_TEMPLATE)
    prompt = prompt_template.format(context=context_text, question=input_text)

    # call LLM model to generate the answer based on the given context and query
    model = ChatOpenAI(temperature=0.0)
    st.info(model.invoke(prompt).content)


# Streamlit UI
# ===============
st.set_page_config(page_title="Contract query/TE-sopimus haku", page_icon=":robot:")
st.header("Contract query/TE-sopimus haku")

form_input = st.text_input(
    "Enter query in the language you want the answer in/ kysy kysymystä kielellä. jolla haluat vastauksen")
submit = st.button("Query/Haku")

if submit:
    st.write(generate_response(form_input))
