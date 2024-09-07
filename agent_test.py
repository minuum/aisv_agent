from langchain.agents import AgentType, initialize_agent
from langchain_community.llms import OpenAI
from langchain.tools import Tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader
from langchain.chains import RetrievalQA

# OpenAI LLM 초기화
llm = OpenAI(temperature=0)

# 문서 로드 및 처리
loader = TextLoader("dummy.txt")
documents = loader.load()
text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(documents)

# 임베딩 및 벡터 저장소 생성
embeddings = OpenAIEmbeddings()
vectorstore = Chroma.from_documents(texts, embeddings)

# RAG 시스템 생성
retriever = vectorstore.as_retriever()
rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

# 도구 정의
tools = [
    Tool(
        name="RAG 시스템",
        func=rag.run,
        description="문서에서 정보를 검색하고 질문에 답변할 때 사용합니다."
    ),
    Tool(
        name="계산기",
        func=lambda x: eval(x),
        description="수학 계산을 수행할 때 사용합니다."
    )
]

# 에이전트 초기화 부분 수정
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5,  # 최대 반복 횟수 설정
    early_stopping_method="generate"  # 조기 종료 방법 설정
)

# 에이전트 실행 부분 수정
prompt = """
문서에 있는 중요한 정보를 분석하고, 다음 사항을 고려하여 상세한 답변을 제공해주세요:
1. 문서의 주요 주제와 핵심 개념
2. 중요한 사실이나 데이터 포인트
3. 이 정보를 바탕으로 도출할 수 있는 주요 결론
4. 가능한 경우, 추가적인 맥락이나 배경 정보
5. 결론의 잠재적 영향이나 적용 방안

자세하고 구조화된 답변을 제공해주세요.
"""

response = agent.run(prompt)
print(response)


# Dockerfile 생성
"""
FROM python:3.12

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

CMD ["python", "agent_test.py"]
"""

# requirements.txt 파일 생성
"""
langchain
openai
chromadb
"""

print("Docker 설정이 완료되었습니다. (Python 3.12 버전 사용)")
print("Dockerfile과 requirements.txt 파일을 프로젝트 루트 디렉토리에 생성해주세요.")
print("Docker 이미지를 빌드하려면 다음 명령어를 실행하세요:")
print("docker build -t agent-test .")
print("Docker 컨테이너를 실행하려면 다음 명령어를 실행하세요:")
print("docker run -it --env-file .env agent-test")

# Docker 사용 설명 (Python 3.12 버전)
"""
1. Dockerfile 생성:
   - 위의 Dockerfile 내용을 프로젝트 루트 디렉토리에 'Dockerfile'이라는 이름으로 저장합니다.
   - Python 3.12 이미지를 기반으로 합니다.

2. requirements.txt 생성:
   - 위의 requirements.txt 내용을 프로젝트 루트 디렉토리에 'requirements.txt'라는 이름으로 저장합니다.

3. .env 파일 생성:
   - OpenAI API 키를 포함한 환경 변수를 .env 파일에 저장합니다.
   예: OPENAI_API_KEY=your_api_key_here

4. Docker 이미지 빌드:
   - 터미널에서 프로젝트 디렉토리로 이동한 후 다음 명령어를 실행합니다:
     docker build -t agent-test .

5. Docker 컨테이너 실행:
   - 다음 명령어로 Docker 컨테이너를 실행합니다:
     docker run -it --env-file .env agent-test

이 설정을 통해 코드를 Python 3.12 기반의 Docker 환경에서 실행할 수 있습니다. 
환경 변수와 의존성이 올바르게 설정되어 있는지 확인하세요.
Python 3.12는 최신 버전이므로 일부 라이브러리가 아직 완전히 호환되지 않을 수 있습니다. 
문제가 발생하면 Python 3.11 또는 3.10으로 다운그레이드하는 것을 고려해보세요.
"""




