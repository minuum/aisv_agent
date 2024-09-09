
from langchain_community.agents import AgentType, initialize_agent
from langchain_community.llms import OpenAI
from langchain.tools import Tool
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import Chroma
from langchain.text_splitter import CharacterTextSplitter
from langchain.document_loaders import TextLoader, PyPDFLoader, YoutubeLoader
from langchain.chains import RetrievalQA
from langchain_community.document_loaders import NotionDBLoader
from langchain_community.utilities import YouTubeSearchAPIWrapper

# OpenAI LLM 초기화
llm = OpenAI(temperature=0)

# 문서 로드 및 처리 함수
def load_and_process_documents(loader):
    documents = loader.load()
    text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
    texts = text_splitter.split_documents(documents)
    return texts

# 임베딩 및 벡터 저장소 생성
embeddings = OpenAIEmbeddings()

# Notion 데이터베이스 로더
notion_loader = NotionDBLoader(
    integration_token="your_notion_integration_token",
    database_id="your_database_id",
    request_timeout_sec=30
)
notion_texts = load_and_process_documents(notion_loader)
notion_vectorstore = Chroma.from_documents(notion_texts, embeddings)

# PDF 로더
pdf_loader = PyPDFLoader("path/to/your/document.pdf")
pdf_texts = load_and_process_documents(pdf_loader)
pdf_vectorstore = Chroma.from_documents(pdf_texts, embeddings)

# YouTube 로더
youtube_loader = YoutubeLoader.from_youtube_url("https://www.youtube.com/watch?v=your_video_id", add_video_info=True)
youtube_texts = load_and_process_documents(youtube_loader)
youtube_vectorstore = Chroma.from_documents(youtube_texts, embeddings)

# RAG 시스템 생성
notion_retriever = notion_vectorstore.as_retriever()
pdf_retriever = pdf_vectorstore.as_retriever()
youtube_retriever = youtube_vectorstore.as_retriever()

notion_rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=notion_retriever)
pdf_rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=pdf_retriever)
youtube_rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=youtube_retriever)

# YouTube 검색 API 래퍼
youtube_search = YouTubeSearchAPIWrapper()

# 도구 정의
tools = [
    Tool(
        name="Notion 데이터베이스",
        func=notion_rag.run,
        description="Notion 데이터베이스에서 정보를 검색하고 질문에 답변할 때 사용합니다."
    ),
    Tool(
        name="PDF 문서",
        func=pdf_rag.run,
        description="PDF 문서에서 정보를 검색하고 질문에 답변할 때 사용합니다."
    ),
    Tool(
        name="YouTube 동영상",
        func=youtube_rag.run,
        description="YouTube 동영상 내용에서 정보를 검색하고 질문에 답변할 때 사용합니다."
    ),
    Tool(
        name="YouTube 검색",
        func=youtube_search.run,
        description="YouTube에서 관련 동영상을 검색할 때 사용합니다."
    ),
    Tool(
        name="계산기",
        func=lambda x: eval(x),
        description="수학 계산을 수행할 때 사용합니다."
    )
]

# 에이전트 초기화
agent = initialize_agent(
    tools, 
    llm, 
    agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
    verbose=True,
    max_iterations=5,
    early_stopping_method="generate"
)

# 에이전트 실행
prompt = """
다음 소스들에서 중요한 정보를 분석하고, 상세한 답변을 제공해주세요:
1. Notion 데이터베이스
2. PDF 문서
3. YouTube 동영상

다음 사항을 고려하여 답변해주세요:
1. 각 소스의 주요 주제와 핵심 개념
2. 중요한 사실이나 데이터 포인트
3. 이 정보를 바탕으로 도출할 수 있는 주요 결론
4. 가능한 경우, 추가적인 맥락이나 배경 정보
5. 결론의 잠재적 영향이나 적용 방안

자세하고 구조화된 답변을 제공해주세요.
"""

response = agent.run(prompt)
print(response)

# requirements.txt에 추가할 내용
"""
langchain
openai
chromadb
pypdf
youtube-transcript-api
notion-client
google-api-python-client
"""

print("새로운 도구가 추가되었습니다. requirements.txt 파일을 업데이트해주세요.")
print("Notion, YouTube, PDF를 사용하기 위한 추가 설정이 필요할 수 있습니다.")
print("각 서비스의 API 키와 필요한 인증 정보를 .env 파일에 추가해주세요.")