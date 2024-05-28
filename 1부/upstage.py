import os
from dotenv import load_dotenv
load_dotenv()
import streamlit as st
import time
import base64
import uuid
import tempfile

os.environ["UPSTAGE_API_KEY"] = os.getenv("UPSTAGE_API_KEY")

from langchain_upstage import UpstageEmbeddings
from langchain_upstage import ChatUpstage
from langchain_chroma import Chroma
from langchain_community.document_loaders import PyPDFLoader

if "id" not in st.session_state:
    st.session_state.id = uuid.uuid4()
    st.session_state.file_cache = {}

session_id = st.session_state.id
client = None

def reset_chat():
    st.session_state.messages = []
    st.session_state.context = None


def display_pdf(file):
    # Opening file from file path

    st.markdown("### PDF Preview")
    base64_pdf = base64.b64encode(file.read()).decode("utf-8")

    # Embedding PDF in HTML
    pdf_display = f"""<iframe src="data:application/pdf;base64,{base64_pdf}" width="400" height="100%" type="application/pdf"
                        style="height:100vh; width:100%"
                    >
                    </iframe>"""

    # Displaying File
    st.markdown(pdf_display, unsafe_allow_html=True)


with st.sidebar:

    st.header(f"Add your documents!")
    
    uploaded_file = st.file_uploader("Choose your `.pdf` file", type="pdf")

    if uploaded_file:
        print(uploaded_file)
        try:
            file_key = f"{session_id}-{uploaded_file.name}"

            with tempfile.TemporaryDirectory() as temp_dir:
                file_path = os.path.join(temp_dir, uploaded_file.name)
                print("file path:", file_path)
                
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getvalue())
                
                file_key = f"{session_id}-{uploaded_file.name}"
                st.write("Indexing your document...")

                if file_key not in st.session_state.get('file_cache', {}):

                    if os.path.exists(temp_dir):
                            print("temp_dir:", temp_dir)
                            loader = PyPDFLoader(
                                file_path
                            )
                    else:    
                        st.error('Could not find the file you uploaded, please check again...')
                        st.stop()

                    pages = loader.load_and_split()

                    vectorstore = Chroma.from_documents(pages, UpstageEmbeddings(model="solar-embedding-1-large"))

                    retriever = vectorstore.as_retriever(k=2)

                    chat = ChatUpstage(upstage_api_key=os.getenv("UPSTAGE_API_KEY"))

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

                st.success("Ready to Chat!")
                display_pdf(uploaded_file)
        except Exception as e:
            st.error(f"An error occurred: {e}")
            st.stop()     

# 웹사이트 제목
st.title("Solar LLM Chatbot")

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
if prompt := st.chat_input("Ask a question!"):
    
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
