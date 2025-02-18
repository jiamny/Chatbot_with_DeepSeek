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
   - Pull the DeepSeek models, for example: deepseek-r1:1.5b 
     
     ollama pull deepseek-r1:1.5b
     

2. **Set Up Python Environment**:
   - Install the required Python packages:
       
     pip install streamlit langchain-ollama
     
3. **Create/edit .streamlit/config.toml file to load images in static folder**:(https://docs.streamlit.io/develop/concepts/configuration/serving-static-files)
   - add the following two line to config.toml and save it:
   
   [server]
   enableStaticServing = true
   
   
## projects

- deepseek_r1_chatbot
- PDF_locally_RAG


## Running the Chatbot

**Start the Streamlit app**:
   - Directly run in IDE like PyCharm
   - or in terminal:
    
   streamlit run deepseek_chat.py
   
