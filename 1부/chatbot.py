# 여기서부터 세줄은 로컬환경에서 돌릴 때에는(즉 웹사이트로 배포 안하고 그냥 터미널에서 돌릴때) 주석처리 해주셔야합니다.
# 배포할때에는 주석처리하시면 안됩니다.
# 주석처리 방법은 "Ctrl + "/"" 누르기
# ---------------------------------------------------
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ---------------------------------------------------

# # Streamlit 배포할때
# # Streamlit 앱의 환경설정에서 꼭 OPENAI_API_KEY = "sk-blabalabla"를 추가해주세요!
# import os
# import streamlit as st
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

import streamlit as st
from dotenv import load_dotenv
load_dotenv()
import openai
import os
openai.api_key= os.environ.get("OPENAI_API_KEY")
import time
from langchain_openai import ChatOpenAI

# 사용할 챗봇 LLM 모델 설정해주기
# temperature는 0에 가까워질수록 형식적인 답변을 내뱉고, 1에 가까워질수록 창의적인 답변을 내뱉음
chat = ChatOpenAI(model="gpt-3.5-turbo-1106", temperature=0.2)

# 1) 데이터 로드하기
from langchain_community.document_loaders import WebBaseLoader

loader = WebBaseLoader("https://dalpha.so/ko/howtouse")
data = loader.load()

# 2) 데이터 쪼개기
from langchain_text_splitters import RecursiveCharacterTextSplitter

# chunk_size는 쪼갤 텍스트의 사이즈를 뜻합니다.
# 쪼개는 이유는 context window가 제한되어 있기 때문.
# 만약 더 큰 데이터 파일이고 chunk_size가 높다면, chunk_overlap = 200으로 설정
# 각 청크 사이에 중요한 맥락을 놓치지 않도록 어느정도 중복되게 하는 것

# RecursiveCharacterTextSplitter: 문서를 \n과 같은 일반적인 구분자를 사용하여 재귀적으로 분할하여 각 청크가 적절한 크기가 될 때까지 분할합니다. 
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=0)
all_splits = text_splitter.split_documents(data)

# 3) 데이터베이스에 업로드하기
from langchain_community.vectorstores import Chroma
from langchain_openai import OpenAIEmbeddings

# 임베딩이라는 뜻은 'similarity search'를 하기 위해서임.
# 각 토큰은 컴퓨터가 이해할 수 있는 숫자인 백터로 바뀌고, 이는 위치와 방향을 갖고 있는 숫자.
# 즉 이 토큰-백터 유사성이 높은 다른 토큰들을 불러오기 위해 임베딩을 해야함.
vectorstore = Chroma.from_documents(documents=all_splits, embedding=OpenAIEmbeddings())

# 4) 데이터를 불러올 수 있는 retriever 만들기 
# k is the number of chunks to retrieve
retriever = vectorstore.as_retriever(k=2)

# 5) 챗봇에 '기억'을 입히기 위한 첫번째 단계 

# 이전의 메시지들과 최신 사용자 질문을 분석해, 문맥에 대한 정보가 없이 혼자서만 봤을때 이해할 수 있도록 질문을 다시 구성함
# 즉 새로 들어온 그 질문 자체에만 집중할 수 있도록 다시 재편성
from langchain.chains import create_history_aware_retriever
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder

contextualize_q_system_prompt = """이전 대화 내용과 최신 사용자 질문이 있을 때, 이 질문이 이전 대화 내용과 관련이 있을 수 있습니다. 
이런 경우, 대화 내용을 알 필요 없이 독립적으로 이해할 수 있는 질문으로 바꾸세요. 
질문에 답할 필요는 없고, 필요하다면 그저 다시 구성하거나 그대로 두세요."""

# MessagesPlaceholder: 'chat_history' 입력 키를 사용하여 이전 메세지 기록들을 프롬프트에 포함시킴.
# 즉 프롬프트, 메세지 기록 (문맥 정보), 사용자의 질문으로 프롬프트가 구성됨. 
contextualize_q_prompt = ChatPromptTemplate.from_messages(
    [
        ("system", contextualize_q_system_prompt),
        MessagesPlaceholder("chat_history"),
        ("human", "{input}"),
    ]
)

# 이를 토대로 메세지 기록을 기억하는 retriever를 생성합니다.
history_aware_retriever = create_history_aware_retriever(
    chat, retriever, contextualize_q_prompt
)

# 두번째 단계로, 방금 전 생성한 체인을 사용하여 문서를 불러올 수 있는 retriever 체인을 생성합니다.
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

# 결과값은 input, chat_history, context, answer 포함함.
rag_chain = create_retrieval_chain(history_aware_retriever, question_answer_chain)

# from langchain_core.messages import HumanMessage

# chat_history = []

# question = "Dalpha AI Store는 어떻게 사용하나요?"
# ai_msg_1 = rag_chain.invoke({"input": question, "chat_history": chat_history})
# chat_history.extend([HumanMessage(content=question), ai_msg_1["answer"]])

# second_question = "그 외 다른 서비스는 어떻게 사용하나요?"
# ai_msg_2 = rag_chain.invoke({"input": second_question, "chat_history": chat_history})

# print(ai_msg_2["answer"])

# 웹사이트 제목
st.title("AI Chatbot")

if "openai_model" not in st.session_state:
    st.session_state["openai_model"] = "gpt-3.5-turbo"

if "messages" not in st.session_state:
    st.session_state.messages = []

# 대화 내용을 기록하기 위해 셋업
# Streamlit 특성상 활성화하지 않으면 내용이 다 날아감.
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        
# 프롬프트 비용이 너무 많이 소요되는 것을 방지하기 위해
MAX_MESSAGES_BEFORE_DELETION = 4

# 웹사이트에서 유저의 인풋을 받고 위에서 만든 AI 에이전트 실행시켜서 답변 받기
if prompt := st.chat_input("Dalpha AI store는 어떻게 사용하나요?"):
    
# 유저가 보낸 질문이면 유저 아이콘과 질문 보여주기
     # 만약 현재 저장된 대화 내용 기록이 4개보다 많으면 자르기
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

        result = rag_chain.invoke({"input": prompt, "chat_history": st.session_state.messages})

        # 증거자료 보여주기
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