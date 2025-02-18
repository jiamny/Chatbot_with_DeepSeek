"""
Streamlit application for PDF-based Retrieval-Augmented Generation (RAG) using Ollama + LangChain.

This application allows users to upload a PDF, process it,
and then ask questions about the content using a selected language model.
"""
from streamlit import runtime
from streamlit.web import cli as stcli
import streamlit as st
import sys
import os
import ollama
import warnings
from typing import Tuple, Any

# Suppress torch warning
warnings.filterwarnings('ignore', category=UserWarning, message='.*torch.classes.*')

from pathlib import Path
from rag_fun import load_and_convert_document, get_markdown_splits, create_or_load_vector_store, build_rag_chain
from langchain_ollama import OllamaEmbeddings
from langchain_community.vectorstores import FAISS
from pdf2image import convert_from_path, exceptions
from PIL import Image

im = Image.open("static/deepseek-color.png")

def main() -> None:
    # Path to vector DB folder
    VECTOR_DB_FOLDER = "vector_db"
    os.makedirs(VECTOR_DB_FOLDER, exist_ok=True)
    selected_model = ""

    # Streamlit page configuration
    st.set_page_config(
        page_title="PDF RAG Streamlit UI",
        page_icon=im,
        layout="wide",
        initial_sidebar_state="collapsed",
    )

    def extract_model_names(models_info: Any) -> Tuple[str, ...]:
        try:
            # The new response format returns a list of Model objects
            if hasattr(models_info, "models"):
                # Extract model names from the Model objects
                model_names = []
                for model in models_info.models:
                    if "deepseek" in model.model:
                        model_names.append(model.model)

            else:
                # Fallback for any other format
                model_names = tuple()
            srt_model_names = sorted( model_names )
            return tuple(srt_model_names)

        except Exception as e:
            return tuple()

    def delete_vector_db() -> None:
        """
        Delete the vector database and clear related session state.
        """
        try:
            # Clear session state
            st.session_state["pdf_pages"] = []
            st.session_state["file_upload"] = ""
            st.session_state["vector_db"] = []
            st.session_state["messages"] = []

            st.success("Collection and temporary files deleted successfully.")
        except Exception as e:
            st.error(f"Error deleting collection: {str(e)}")


    def get_pdf_image_files(pdf_path, file_name):
        pdf_images = []
        try:
            images_folder = Path(VECTOR_DB_FOLDER + "/" + file_name + "/" + "images")
            os.makedirs(images_folder, exist_ok=True)

            # Check if images already exist
            image_paths = list(images_folder.glob("*.png"))
            if image_paths:
                # If images exist, display them
                for img_path in image_paths:
                    image = Image.open(img_path)
                    pdf_images.append(image)
            else:
                # Convert PDF to images (one per page)
                images = convert_from_path(pdf_path)
                for i, image in enumerate(images):
                    img_path = images_folder / f"page_{i + 1}.png"
                    image.save(img_path, "PNG")  # Save image to disk
                    pdf_images.append(image)
            return pdf_images
        except exceptions.PDFPageCountError:
            st.error("Error: Unable to get page count. The PDF may be corrupted or empty.")
        except exceptions.PDFSyntaxError:
            st.error("Error: PDF syntax is invalid or the document is corrupted.")
        except Exception as e:
            st.error(f"Error loading PDF: {str(e)}")

    st.header("🧠 PDF RAG with DeepSeek", divider="gray", anchor=False)

    # Get available models
    models_info = ollama.list()
    available_models = extract_model_names(models_info)

    # Create layout
    col1, col2 = st.columns([2, 3])

    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state["messages"] = []
    if "vector_db" not in st.session_state:
        st.session_state["vector_db"] = []
    if "pdf_pages" not in st.session_state:
        st.session_state["pdf_pages"] = []
    if "file_upload" not in st.session_state:
        st.session_state["file_upload"] = ""

    # Model selection
    if available_models:
        selected_model = col2.selectbox(
            "Pick a model available locally on your system ↓", 
            available_models,
            key="model_select"
        )

    # Dropdown to select vector DB or upload a new document
    vector_db_options = [f.stem for f in Path(VECTOR_DB_FOLDER).glob("*.faiss")]
    vector_db_options.append("Upload New Document")  # Add option to upload a new document
    selected_vector_db = col1.selectbox("Select Vector DB or Upload New Document", vector_db_options, index=0)
    vector_store = None

    # If 'Upload New Document' is selected, show the file uploader
    if selected_vector_db == "Upload New Document":
        #delete_vector_db(st.session_state["vector_db"])
        # Clear session state
        st.session_state["pdf_pages"] = []
        st.session_state["file_upload"] = ""
        st.session_state["vector_db"] = []
        st.session_state["messages"] = []

        uploaded_file = col1.file_uploader("Upload a PDF file for analysis", type=["pdf"])

        # Process the uploaded PDF
        if uploaded_file:
            col1.subheader("Uploaded PDF")
            col1.write(uploaded_file.name)

            # Save the PDF file temporarily and display it
            temp_path = f"temp_{uploaded_file.name}"
            document_binary = uploaded_file.read()
            with open(temp_path, "wb") as f:
                f.write(document_binary)

            # Display PDF in the sidebar (show all pages)
            st.session_state["pdf_pages"] = get_pdf_image_files(temp_path, uploaded_file.name.split('.')[0])
            st.session_state["vector_db"] = [uploaded_file.name.split('.')[0]]

            # PDF processing button
            if col1.button("Process PDF and Store in Vector DB"):
                with col1.spinner("Processing document..."):
                    # Convert PDF to markdown directly
                    markdown_content = load_and_convert_document(temp_path)
                    chunks = get_markdown_splits(markdown_content)

                    # Initialize embeddings
                    embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")

                    # Create or load vector DB and store PDF along with it
                    vector_store = create_or_load_vector_store(uploaded_file.name.split(".")[0], chunks, embeddings)

                    # Ensure vector DB and PDF are stored correctly
                    vector_db_path = Path(VECTOR_DB_FOLDER) / f"{uploaded_file.name.split('.')[0]}.faiss"
                    vector_store.save_local(str(vector_db_path))  # Save FAISS vector store

                    # Store the PDF file alongside the vector DB
                    pdf_path = Path(VECTOR_DB_FOLDER) / f"{uploaded_file.name}"
                    with open(pdf_path, "wb") as f:
                        f.write(document_binary)

                    st.success("PDF processed and stored in the vector database.")

                    # Clean up the temporary file
                    Path(temp_path).unlink()

    elif selected_vector_db != "Upload New Document":
        # Load the selected vector DB
        vector_db_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.faiss"
        if vector_db_path.exists():
            embeddings = OllamaEmbeddings(model='nomic-embed-text', base_url="http://localhost:11434")
            vector_store = FAISS.load_local(str(vector_db_path), embeddings=embeddings,
                                            allow_dangerous_deserialization=True)
            # Display PDF in the sidebar
            pdf_path = Path(VECTOR_DB_FOLDER) / f"{selected_vector_db}.pdf"
            if pdf_path.exists():
                st.session_state["pdf_pages"] = get_pdf_image_files(pdf_path, selected_vector_db)
            else:
                col1.warning("PDF file not found for the selected vector DB.")
            st.session_state["vector_db"] = [selected_vector_db]
        else:
            col1.warning(f"Vector DB '{selected_vector_db}' not found.")

    # Display PDF if pages are available
    if "pdf_pages" in st.session_state and st.session_state["pdf_pages"]:
        # PDF display controls
        zoom_level = col1.slider(
            "Zoom Level", 
            min_value=100, 
            max_value=1000, 
            value=700, 
            step=50,
            key="zoom_slider"
        )

        # Display PDF pages
        with col1:
            with st.container(height=410, border=True):
                for page_image in st.session_state["pdf_pages"]:
                    st.image(page_image, width=zoom_level)

    # Delete collection button
    delete_collection = col1.button(
        "🧹 Delete collection",
        type="secondary",
        key="delete_button"
    )

    if delete_collection:
        delete_vector_db()


    # Chat interface
    with (col2):
        message_container = st.container(height=500, border=True)
        
        # Display chat history
        if st.session_state["messages"]:
            for i, message in enumerate(st.session_state["messages"]):
                avatar = "🤖" if message["role"] == "assistant" else "😎"
                with message_container.chat_message(message["role"], avatar=avatar):
                    if message["role"] == "assistant":
                        response = message["content"]
                        response_placeholder = st.empty()
                        response_placeholder.markdown("""
                            <div style="height:320px; overflow-y:auto">""" +
                                response +
                        """</div>""", unsafe_allow_html=True)
                    else:
                        st.markdown(message["content"])

        # Chat input and processing
        if prompt := st.chat_input(
                "Enter your question: i.e., What is the company's revenue for the quarter?",
                key="chat_input"):

            if st.session_state["vector_db"]:
                try:
                    # Add user message to chat
                    st.session_state["messages"].append({"role": "user", "content": prompt})
                    with message_container.chat_message("user", avatar="😎"):
                        st.markdown(prompt)

                    # Process and display assistant response
                    with message_container.chat_message("assistant", avatar="🤖"):
                        response = ""
                        with st.spinner(":green[processing...]"):
                            if st.session_state["vector_db"]:
                                retriever = vector_store.as_retriever(search_type="mmr", search_kwargs={'k': 5})
                                response_placeholder = st.empty()
                                # Build and run the RAG chain
                                rag_chain = build_rag_chain(retriever, selected_model)
                                for chunk in rag_chain.stream(prompt):
                                    response += chunk
                                    response_placeholder.markdown("""
                                        <div style="height:320px; overflow-y:auto">""" +
                                            response.replace('$', '\\$') +
                                        """</div>""", unsafe_allow_html=True)
                            else:
                                st.warning("Please upload a PDF file first.")

                        # Add assistant response to chat history
                        if response != "":
                            st.session_state["messages"].append(
                                {"role": "assistant", "content": response}
                            )

                except Exception as e:
                    st.error(e, icon="⛔️")

            if not st.session_state["vector_db"]:
                st.warning("Upload a PDF file or use the sample PDF to begin chat...")


if __name__ == "__main__":
    if runtime.exists():
        main()
    else:
        sys.argv = ["streamlit", "run", sys.argv[0]]
        sys.exit(stcli.main())
