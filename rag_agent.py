import os
from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.document_loaders import (
    PyPDFLoader,
    UnstructuredExcelLoader,
    CSVLoader,
    UnstructuredWordDocumentLoader,
    TextLoader,
)
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain.tools import Tool
from typing import TypedDict, Annotated, Sequence
from langchain_core.messages import HumanMessage, BaseMessage, AIMessage, SystemMessage, ToolMessage
from operator import add as add_messages
from langgraph.graph import StateGraph, END
from textblob import TextBlob
# import matplotlib.pyplot as plt
from collections import defaultdict
from tempfile import NamedTemporaryFile
import streamlit as st


load_dotenv()

chat_llm = ChatGoogleGenerativeAI(model = 'gemini-2.0-flash',temperature = 0.1)

embedding = GoogleGenerativeAIEmbeddings(model="models/embedding-001")

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size = 1000,
    chunk_overlap = 100
)

persist_directory = "chroma_db"
collection_name = "user_file"

st.title("📄 Document Q&A with RAG Agent")

uploaded_file = st.file_uploader("Upload your document please", type=['pdf','txt','csv','docx','xlsx'])
query = st.chat_input("Ask a question about the document:")

if uploaded_file:

    provided_document = os.path.splitext(uploaded_file.name)[0]
    file_ext = os.path.splitext(uploaded_file.name)[1].lower()

    with NamedTemporaryFile(delete=False, suffix=file_ext) as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name


    if file_ext == ".pdf":
        loader = PyPDFLoader(temp_path)
    elif file_ext in [".xls", ".xlsx"]:
        loader = UnstructuredExcelLoader(temp_path)
    elif file_ext == ".csv":
        loader = CSVLoader(temp_path)
    elif file_ext in [".doc", ".docx"]:
        loader = UnstructuredWordDocumentLoader(temp_path)
    elif file_ext == ".txt":
        loader = TextLoader(temp_path)
    else:
        st.error("Unsupported file format.")
        st.stop()

    try:
        pages = loader.load()
        st.success(f"File has been loaded and has {len(pages)} pages")
    except Exception as e:
        st.error(f"Error loading file: {e}")
        st.stop()

    page_split = text_splitter.split_documents(pages)

    try: 
        vectorStore = Chroma.from_documents(
            persist_directory=persist_directory,
            collection_name=collection_name,
            documents=page_split,
            embedding=embedding
        )
        print(f"Database VectorStore Created Successfully")
    except Exception as e:
        st.error("⚠️ Failed to build vector DB.")
        st.exception(e)
        st.stop()

    retriever = vectorStore.as_retriever(
        search_type="similarity",
        search_kwargs={'k': 5}
    )

    def retriever_tool_func(query:str) -> str:
        docs = retriever.invoke(query)

        if not docs:
            return "I Found No Relevant Information In The Provided File"
        
        return '\n\n'.join(f"Document {i+1}:\n{doc.page_content}" for i,doc in enumerate(docs))

    retriever_tool = Tool.from_function(
        name = 'retriever_tool',
        func = retriever_tool_func,
        description=f"Search and return required information from the {provided_document} document. Input should be the user query."
    )

    tools = [retriever_tool]

    llm = chat_llm.bind_tools(tools)

    class AgentState(TypedDict):
        messages: Annotated[Sequence[BaseMessage],add_messages]

    def should_continue(state:AgentState):
        """Check if the last message contain tool message"""
        result = state['messages'][-1]
        tool_calls = getattr(result, 'tool_calls', None)
        return tool_calls is not None and len(tool_calls) > 0

    system_prompt = f"""
    You are an intelligent AI assistant who answers questions about {provided_document} document loaded into your knowleage. 
    Use the retriever tool available to answer questions about the Resume provided to you. You can make multiple calls if needed.
    If you need to look up some information before asking a follow up question, you are allowed to do that!
    Please always cite the specific parts of the documents you use in your answers.
    """

    tools_dict = {our_tool.name: our_tool for our_tool in tools}

    def call_llm(state: AgentState) -> AgentState:
        """Function to call the llm from the current state."""
        messages = [SystemMessage(content=system_prompt)] + list(state['messages'])
        message = llm.invoke(messages)
        return {'messages': [message]}

    def take_action(state: AgentState) -> AgentState:
        """Execute tool calls from the LLM's response."""

        tool_calls = getattr(state['messages'][-1], 'tool_calls', None)
        results = []

        if tool_calls is not None:
            for t in tool_calls:
                print(f"Calling Tool: {t['name']} with query : {t['args'].get('query', 'No query provided')}")

                if not t['name'] in tools_dict:
                    print(f"\nTool: {t['name']} does not exist.")
                    result = "Incorrect Tool Name, Please Retry and Select tool from List of Avaiable tools."

                else:
                    result = tools_dict[t['name']].invoke(t['args'].get('query',''))
                    print(f"Result length: {len(str(result))}")

                results.append(ToolMessage(tool_call_id=t["id"],name=t['name'],content=str(result)))

        print("Tool Execution Completed. Back to the model!")
        return {'messages':results}

    graph = StateGraph(AgentState)
    graph.add_node("llm",call_llm)
    graph.add_node("retriver_agent", take_action)

    graph.add_conditional_edges(
        "llm",
        should_continue,
        {True:"retriver_agent",False:END}
    )

    graph.add_edge("retriver_agent","llm")
    graph.set_entry_point("llm")

    rag_agent = graph.compile()

    if "chat_history" not in st.session_state:
        st.session_state.chat_history = []

    # Display previous messages
    for msg in st.session_state.chat_history:
        if isinstance(msg, HumanMessage):
            st.chat_message("user").write(msg.content)
        elif isinstance(msg, AIMessage):
            st.chat_message("ai").write(msg.content)
        # elif isinstance(msg, ToolMessage):
        #     st.chat_message("assistant").write(f"🔧 Tool response:\n{msg.content}")

    if query:
        # Add and display the question immediately
        human_message = HumanMessage(content=query)
        st.session_state.chat_history.append(human_message)
        st.chat_message("user").write(query)
        
        # Show a spinner while generating response
        with st.spinner("AI is thinking..."):
            response = rag_agent.invoke({"messages": st.session_state.chat_history})
            st.session_state.chat_history = response['messages']
        
        # Display the response after loading is complete
        st.chat_message("ai").write(response['messages'][-1].content)