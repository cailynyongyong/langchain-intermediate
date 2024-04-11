# 필요한 패키지
# %pip install --upgrade --quiet  google-api-python-client > /dev/null
# %pip install --upgrade --quiet  google-auth-oauthlib > /dev/null
# %pip install --upgrade --quiet  google-auth-httplib2 > /dev/null
# %pip install --upgrade --quiet  beautifulsoup4 > /dev/null # This is optional but is useful for parsing HTML messages
# pip install -U langchainhub

from langchain_community.agent_toolkits import GmailToolkit
from dotenv import load_dotenv
load_dotenv()
import openai
import os
openai.api_key= os.environ.get("OPENAI_API_KEY")

toolkit = GmailToolkit()

from langchain_community.tools.gmail.utils import (
    build_resource_service,
    get_gmail_credentials,
)

# Can review scopes here https://developers.google.com/gmail/api/auth/scopes
# For instance, readonly scope is 'https://www.googleapis.com/auth/gmail.readonly'
credentials = get_gmail_credentials(
    token_file="token.json",
    scopes=["https://mail.google.com/"],
    client_secrets_file="credentials.json",
)
api_resource = build_resource_service(credentials=credentials)
toolkit = GmailToolkit(api_resource=api_resource)

tools = toolkit.get_tools()
print(tools)

from langchain import hub
from langchain.agents import AgentExecutor, create_openai_functions_agent
from langchain_openai import ChatOpenAI

instructions = """You are an assistant."""
base_prompt = hub.pull("langchain-ai/openai-functions-template")
prompt = base_prompt.partial(instructions=instructions)

llm = ChatOpenAI(temperature=0)

agent = create_openai_functions_agent(llm, toolkit.get_tools(), prompt)

agent_executor = AgentExecutor(
    agent=agent,
    tools=toolkit.get_tools(),
    verbose=False,
)

result = agent_executor.invoke(
    {
        "input": "내 메일함에서 '이력서'라는 제목의 가장 최근 이메일을 찾아주세요."
                "임시보관함이 아닌, 받은 편지함에서만 검색해주세요."
                "이 이메일을 기반으로 Gmail 초안을 만들어 이 이메일에 답장해주세요. 어떤 경우에도 메시지를 발송해서는 안 됩니다."
    }
)

print(result)