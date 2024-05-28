import os
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler
from langchain_openai import ChatOpenAI
# from langchain import OpenAI, ConversationChain, LLMChain, PromptTemplate
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
from langchain.chains.conversation.memory import ConversationBufferWindowMemory
import dotenv 
dotenv.load_dotenv()

# Initializes your app with your bot token and socket mode handler
app = App(token=os.environ.get("SLACK_BOT_TOKEN"))

llm = ChatOpenAI(model="gpt-3.5-turbo", temperature=0)

# 검색 엔진 연동하기
from langchain_community.tools.tavily_search import TavilySearchResults
search = TavilySearchResults()

# 에이전트가 사용할 수 있는 툴 만들기
tools = [search]

# 프롬프트 만들어주기
from langchain import hub
from langchain.agents import create_openai_functions_agent
from langchain.agents import AgentExecutor

instructions = """어시스턴트는 단순한 질문에 답하는 것부터 광범위한 주제에 대해 심도 있는 설명과 토론을 제공하는 등 다양한 작업을 돕도록 설계되었습니다. 언어 모델로서 어시스턴트는 받은 입력을 기반으로 인간처럼 텍스트를 생성할 수 있으며, 자연스럽게 들리는 대화를 진행하고 관련된 주제에 대해 일관성 있고 적절한 응답을 제공할 수 있습니다.

어시스턴트는 지속적으로 학습하고 개선되고 있으며, 그 능력은 끊임없이 진화하고 있습니다. 대량의 텍스트를 처리하고 이해할 수 있으며, 이 지식을 활용하여 다양한 질문에 정확하고 유익한 응답을 제공할 수 있습니다. 또한, 어시스턴트는 받은 입력을 기반으로 자체적인 텍스트를 생성할 수 있어 다양한 주제에 대한 토론을 진행하고 설명 및 설명을 제공할 수 있습니다.

전반적으로 어시스턴트는 다양한 작업에 도움을 줄 수 있는 강력한 도구이며, 광범위한 주제에 대해 가치 있는 통찰과 정보를 제공할 수 있습니다. 특정 질문에 대한 도움이 필요하든, 특정 주제에 대해 대화를 나누고 싶든, 어시스턴트는 도와줄 준비가 되어 있습니다.

꼭 한국어로 대답해주세요."""

# 프롬프트 템플릿 확인하기: https://smith.langchain.com/hub/hwchase17/openai-functions-agent
base_prompt = hub.pull("hwchase17/openai-functions-agent")
prompt = base_prompt.partial(instructions=instructions)

# 에이전트 활성화하기
agent = create_openai_functions_agent(llm, tools, prompt)
agent_executor = AgentExecutor(agent=agent, tools=tools, verbose=False)

# agent_executor.invoke({"input": "한국의 날씨가 어떤가요?"})

#Event handler for Slack
@app.event("app_mention")
def handle_app_mention_events(body, say, logger):
    message = body["event"]['text']

    output = agent_executor.invoke({"input": message})  
    say(output) 

#Message handler for Slack
@app.message(".*")
def message_handler(message, say, logger):
    print(message)
    
    output = agent_executor.invoke({"input": message['text']})   
    say(output['output'])

# Start your app
if __name__ == "__main__":
    SocketModeHandler(app, os.environ["SLACK_APP_TOKEN"]).start()