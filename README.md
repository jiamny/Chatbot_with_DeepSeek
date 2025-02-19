# DeepSeek Chatbot with Ollama and LangChain

Fully functional, locally run chatbot and pdf RAG powered by **DeepSeek LLM**, **Ollama**, and **LangChain**. Plus attractive Streamlit-based front-end with chat history and modern UI.


## Prerequisites

Before running the chatbot, ensure you have the following installed:

1. **Python 3.10+**
2. **Ollama** 
3. **Streamlit** 
4. **LangChain** 


## Installation

1. **Install Ollama**:
   - Download and install Ollama from [here](https://ollama.ai/).
   - Pull embedding model nomic-embed-text, and the DeepSeek models, like deepseek-r1:1.5b 
     
     ollama pull deepseek-r1:1.5b
     ollama pull deepseek-r1:latest
     ollama pull nomic-embed-text 
     

2. **Set Up Python Environment**:
   - Install the required Python packages:
       
     pip install -r requirements.txt
     
3. **Create/edit .streamlit/config.toml file to load local images**:(https://docs.streamlit.io/develop/concepts/configuration/serving-static-files)
   - add the following two line to config.toml and save it:
   
   [server]
   enableStaticServing = true
   
   
## projects
1. **deepseek_r1_chatbot** 
![Alt text](./deep_seek_chat.png?raw=true)

2. **PDF_locally_RAG**
![Alt text](./pdf_rag.png?raw=true)


## Running the Chatbot

**Start the Streamlit app**:
   - Directly run in IDE like PyCharm
   - or in terminal:
    
   streamlit run deepseek_chat.py
   
