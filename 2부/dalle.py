# 여기서부터 세줄은 로컬환경에서 돌릴 때에는(즉 웹사이트로 배포 안하고 그냥 터미널에서 돌릴때) 주석처리 해주셔야합니다. 
# 배포할때에는 주석처리하시면 안됩니다. 
# 주석처리 방법은 "Ctrl + "/"" 누르기
# ---------------------------------------------------
# __import__('pysqlite3')
# import sys
# sys.modules['sqlite3'] = sys.modules.pop('pysqlite3')
# ---------------------------------------------------

# Streamlit 배포하고 싶다면
# Streamlit 앱의 환경설정에서 꼭 OPENAI_API_KEY = "sk-blabalabla"를 추가해주세요!
# os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]

# 필요한 모듈들 불러오기 
import streamlit as st
from langchain_community.llms import OpenAI
import requests
from io import BytesIO

st.set_page_config(page_title="DALL-E Chatbot", page_icon="🌠")
st.title("DALLE Chatbot")

# 로컬 환경에서 테스트해볼때
from dotenv import load_dotenv
load_dotenv()
import openai
import os
openai.api_key= os.environ.get("OPENAI_API_KEY")

# OpenAI LLM 셋업하기. temperature = 0.9는 더 창의적인 프롬프트를 생성하라는 뜻.
llm = OpenAI(temperature=0.9)

from openai import OpenAI
client = OpenAI()

# Initialize or get the existing chat history from the session state
if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

if user_query := st.chat_input(placeholder="Design a greeting card for Christmas"):
    st.session_state.messages.append({"role": "user", "content": user_query})
    with st.chat_message("user"):
        st.markdown(user_query)

    # 생성된 이미지 보여주기!
    with st.chat_message("assistant"):
        # DALLE 3 사용할 경우 (한개의 이미지만 생성 가능)
        response = client.images.generate(
            model="dall-e-3",
            prompt=user_query,
            size="1024x1024",
            quality="standard",
            n=1,
        )

        image_url = response.data[0].url
        st.image(image_url)

        response = requests.get(image_url)
        if response.status_code == 200:
            # Prepare the file to download
            img_bytes = BytesIO(response.content)
            img_bytes.name = 'downloaded_image.jpg'  # or the appropriate file extension
            st.download_button(
                label="이미지 다운로드",
                data=img_bytes,
                file_name=img_bytes.name,
                mime="image/jpeg"  # or the appropriate MIME type
            )
        else:
            st.error("Failed to download the image.")

        st.session_state.messages.append({"role": "assistant", "content": image_url})

    #     # DALLE 2 사용할 경우 (여러개의 이미지를 만들 수 있음. n=2 개수만큼 만들어줌.)
    #     response = client.images.generate(
    #         model="dall-e-2",
    #         prompt=user_query,
    #         size="1024x1024",
    #         quality="standard",
    #         n=2,
    #     )

    #     image_url = response.data[0].url
    #     image_url2 = response.data[1].url

    #     col1, col2 = st.columns(2)
    #     with col1:
    #         st.image(image_url)
    #         response = requests.get(image_url)
    #         if response.status_code == 200:
    #             # Prepare the file to download
    #             img_bytes = BytesIO(response.content)
    #             img_bytes.name = 'downloaded_image.jpg'  # or the appropriate file extension
    #             st.download_button(
    #                 label="이미지 다운로드",
    #                 data=img_bytes,
    #                 file_name=img_bytes.name,
    #                 mime="image/jpeg"  # or the appropriate MIME type
    #             )
    #         else:
    #             st.error("Failed to download the image.")

    #     with col2:
    #         st.image(image_url2)
    #         response = requests.get(image_url2)
    #         if response.status_code == 200:
    #             # Prepare the file to download
    #             img_bytes = BytesIO(response.content)
    #             img_bytes.name = 'downloaded_image2.jpg'  # or the appropriate file extension
    #             st.download_button(
    #                 label="이미지 다운로드",
    #                 data=img_bytes,
    #                 file_name=img_bytes.name,
    #                 mime="image/jpeg"  # or the appropriate MIME type
    #             )
    #         else:
    #             st.error("Failed to download the image.")

    # st.session_state.messages.append({"role": "assistant", "content": [image_url, image_url]})


