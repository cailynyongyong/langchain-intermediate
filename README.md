## Langchain 자료

### 환경설정

Make sure to fill in the API keys for OpenAI.

```
python -m venv venv
source venv/bin/activate
pip install --upgrade --quiet  langchain langchain-community langchainhub langchain-openai chromadb bs4 python-dotenv openai
```

If streamlit is not working, checkout their [installation page](https://docs.streamlit.io/library/get-started/installation)

requirements.txt 다운받는 방법

```
pip install -r /path/to/requirements.txt
```

requirements.txt 만드는 방법

```
pip3 freeze > requirements.txt  # Python3
pip freeze > requirements.txt  # Python2
```
