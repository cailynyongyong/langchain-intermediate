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
    # This is set to False to prevent information about my email showing up on the screen
    # Normally, it is helpful to have it set to True however.
    verbose=False,
)

result = agent_executor.invoke(
    {
        "input": "Search in my inbox the latest email with the subject '이력서'."
                "Do not search in the drafts folder, only in the received inbox folder."
                "Based on this email, please create a gmail draft that responds to this email. Under no circumstances may you send the message."
    }
)

print(result)