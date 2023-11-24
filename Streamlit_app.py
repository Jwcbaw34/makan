import os
import streamlit as st
import requests 
from langchain.document_loaders import PyMuPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chat_models import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.chains import RetrievalQA
from htmlTemplates import css, bot_template, user_template
from langchain import PromptTemplate

# Call set_page_config here at the top-level scope
st.set_page_config(page_title="Makan Inspiration :bulb:", page_icon=":spoon:")

openai_api_key = os.environ.get("OPENAI_API_KEY")
PASSWORD = "FHS"

# Define the URL of your document on GitHub
github_document_url = "https://raw.githubusercontent.com/Jwcbaw34/makan/main/Food%20options.pdf"

# Check if the document is already downloaded; if not, download it
local_document_path = "Food options.pdf"

if not os.path.exists(local_document_path):
    st.info("Downloading document...")
    response = requests.get(github_document_url)

    if response.status_code == 200:
        with open(local_document_path, "wb") as file:
            file.write(response.content)
        st.success(" ")
    else:
        st.error("error")
else:
    st.success("It's that time of the day again, when no one can decide where to eat. Makan Mama is here to help!")

# Define the directory where you want to store the vector store
VECTORSTORE_PATH = "vectorstore"  # You can customize this path if needed


def handler_verify():
    input_password = st.session_state.password_input
    if input_password == PASSWORD:
        st.session_state.password_flag = True
    else:
        st.write("Incorrect Password")

def initialize_app():
    st.text_input(
        label = "Enter Password",
        key = "password_input",
        type = "password",
        on_change = handler_verify,
        placeholder = "Enter Password",
        )

# If the vector store directory doesn't exist, create it
if not os.path.exists(VECTORSTORE_PATH):

    # Load and process the document
    document = PyMuPDFLoader(local_document_path).load()

    # Process each page in the document (assuming 'document' is a list of pages)
    texts = []
    for page in document:
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
        page_texts = text_splitter.split_documents([page])
        texts.extend(page_texts)

    # Create embeddings
    embeddings = OpenAIEmbeddings()

    # Create the vector store
    docsearch = FAISS.from_documents(texts, embeddings)
    docsearch.save_local(VECTORSTORE_PATH)

def get_qasource_chain(docsearch):
    qasource_chain = RetrievalQA.from_chain_type(
        llm=ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0, max_tokens=1024),
        chain_type="stuff",
        retriever=docsearch.as_retriever(),
        chain_type_kwargs={
            "verbose": True,
        },
        return_source_documents=True,
        verbose=True
    )
    return qasource_chain

def handle_userinput(response):
    # Add user's question and bot's response to the chat history
    st.write(bot_template.replace("{{MSG}}", response['result']), unsafe_allow_html=True)

def main():
    if "password_flag" not in st.session_state:
        st.session_state.password_flag = False

    docsearch = FAISS.load_local(VECTORSTORE_PATH, OpenAIEmbeddings())
    qasource_chain = get_qasource_chain(docsearch)

    if st.session_state.password_flag:
        st.header("Makan Mama :spoon:")

        # Add new input fields
        selected_time = st.selectbox('How much time do you have?', ['30 min', '1 hr', '>1 hr'])
        is_raining = st.selectbox('Is it raining?', ['Yes', 'No'])
        mood = st.selectbox('How are you feeling?', ['adventurous', 'stressed', 'at peace', 'tired', 'resolute'])
        additional_requirements = st.text_area('Anything else I should know? Special needs / wishes / cravings?')

        if st.button("Receive Makan Mama's decision"):
            # Construct context from new inputs
            user_context = f"Time available: {selected_time}. Raining: {is_raining}. Additional requirements: {additional_requirements}. Mood: {mood}."
            
            response = qasource_chain({"query": user_context})  # Use qasource_chain with the full query
            handle_userinput(response)  # Pass the response to handle_userinput()
            

    else:
        initialize_app()

if __name__ == "__main__":
    main()
