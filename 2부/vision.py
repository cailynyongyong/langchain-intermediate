import streamlit as st
import base64
import requests
from dotenv import load_dotenv
load_dotenv()
import openai
import os
openai.api_key= os.environ.get("OPENAI_API_KEY")
from langchain_openai import ChatOpenAI
import time

chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

from langchain_community.document_loaders import WebBaseLoader

# loader = WebBaseLoader("https://dalpha.so/ko/howtouse")
# data = loader.load()
# print("web data: ", data)

# 이번에는 url이 아닌 txt 파일 업로드
with open("manual.txt", 'r', encoding='utf-8') as f:
    data = f.read()

from langchain_text_splitters import CharacterTextSplitter, RecursiveCharacterTextSplitter

text_splitter = RecursiveCharacterTextSplitter.from_tiktoken_encoder(
 chunk_size=500, chunk_overlap=0
)
all_splits = text_splitter.split_text(data)

from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

vectorstore = Chroma.from_texts(texts=all_splits, embedding=OpenAIEmbeddings())

# k is the number of chunks to retrieve
retriever = vectorstore.as_retriever(k=4)

from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 
이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 
질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)
history_aware_retriever = create_history_aware_retriever(
    chat, retriever, contextualize_q_prompt
)

from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain

qa_system_prompt = """질문-답변 업무를 돕는 보조원입니다. 
질문에 답하기 위해 검색된 내용을 사용하세요. 
답을 모르면 모른다고 말하세요. 
답변은 세 문장 이내로 간결하게 유지하세요.

{context}"""
qa_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", qa_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

question_answer_chain = create_stuff_documents_chain(chat, qa_prompt)

rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# 업로드한 이미지를 인코딩하기 위한 함수
def encode_image(image_file):
    # 이미지 파일을 읽어 base64 문자열로 인코딩한다.
    return base64.b64encode(image_file.read()).decode('utf-8')

st.title("이미지 인식 챗봇")
if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# Streamlit 웹사이트에서 파일 업로드   
# 이미지 경로     
uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])

if uploaded_file is not None:
    # base64 인코딩된 이미지 문자열을 얻는다
    base64_image = encode_image(uploaded_file)

    # OpenAI API Key
    api_key = os.environ.get("OPENAI_API_KEY")

    # HTTP 헤더 설정
    headers = {
      "Content-Type": "application/json",
      "Authorization": f"Bearer {api_key}"
    }

    # API 요청 본문
    payload = {
      "model": "gpt-4-turbo",
      "messages": [
        {
          "role": "user",
          "content": [
            {
              "type": "text",
              "text": "What’s in this image?"
            },
            {
              "type": "image_url",
              "image_url": {
                "url": f"data:image/jpeg;base64,{base64_image}"
              }
            }
          ]
        }
      ],
      "max_tokens": 300
    }

    # POST 요청을 통해 OpenAI API에 접근하고 응답을 받는다.
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)

    # 이미지 보여주기
    data = response.json()
    image_explanation = data['choices'][0]['message']['content']
    st.image(f"data:image/jpeg;base64,{base64_image}")

    # 이미지에 대한 설명 대화 기록에 저장하기
    st.session_state.messages.append({"role": "assistant", "content": image_explanation})
    print("after image session state: ", st.session_state.messages)

# Define a threshold for the maximum number of messages before deletion occurs
MAX_MESSAGES_BEFORE_DELETION = 4

# 웹사이트에서 유저의 인풋을 받고 위에서 만든 AI 에이전트 실행시켜서 답변 받기
if prompt := st.chat_input("Ask question"):
    
     # Check if the number of messages exceeds the threshold
    if len(st.session_state.messages) >= MAX_MESSAGES_BEFORE_DELETION:
        # Remove the first two messages
        del st.session_state.messages[0]
        del st.session_state.messages[0]  
   
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

# AI가 보낸 답변이면 AI 아이콘이랑 LLM 실행시켜서 답변 받고 스트리밍해서 보여주기
    with st.chat_message("assistant"):
        message_placeholder = st.empty()
        full_response = ""

        # 이미지에 있는 설명이 대화 기록에 저장됨. 이 설명을 바탕으로 업로드한 데이터파일에 대해 검색할 수 있음. 
        result = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})

        with st.expander("Evidence context"):
            st.write(result["context"])

        for chunk in result["answer"].split(" "):
            full_response += chunk + " "
            time.sleep(0.2)
            message_placeholder.markdown(full_response + "▌")
            message_placeholder.markdown(full_response)
            
    st.session_state.messages.append({"role": "assistant", "content": full_response})

print("_______________________")
print(st.session_state.messages)