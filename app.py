import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain_openai import OpenAI
from langchain.tools import Tool
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader, PyPDFLoader, YoutubeLoader, NotionDBLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
from langchain_community.tools.youtube import search
import os
import tempfile
from PyPDF2 import PdfReader
from io import BytesIO
from langchain_community.vectorstores.utils import filter_complex_metadata
from langchain.schema import Document

# Streamlit 앱 제목 설정
st.title("LangChain RAG 시스템 및 에이전트")

# API 키 입력
api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
notion_token = st.text_input("Notion 통합 토큰을 입력하세요:", type="password")
youtube_api_key = st.text_input("YouTube API 키를 입력하세요:", type="password")

os.environ["OPENAI_API_KEY"] = api_key
os.environ["NOTION_TOKEN"] = notion_token
os.environ["YOUTUBE_API_KEY"] = youtube_api_key

# 입력 소스 선택
source_type = st.selectbox("입력 소스를 선택하세요:", ["텍스트", "PDF", "YouTube", "Notion"])

if source_type == "텍스트":
    doc_content = st.text_area("분석할 문서 내용을 입력하세요:")
elif source_type == "PDF":
    uploaded_file = st.file_uploader("PDF 파일을 업로드하세요", type="pdf")
    if uploaded_file is not None:
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
                tmp_file.write(uploaded_file.getvalue())
                tmp_file_path = tmp_file.name

            pdf_loader = PyPDFLoader(tmp_file_path)
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_documents(pdf_loader.load())
        except Exception as e:
            st.error(f"PDF 파일 처리 중 오류가 발생했습니다: {str(e)}")
        finally:
            if 'tmp_file_path' in locals():
                os.unlink(tmp_file_path)  # 임시 파일 삭제
elif source_type == "YouTube":
    youtube_url = st.text_input("YouTube 동영상 URL을 입력하세요:")
elif source_type == "Notion":
    notion_db_id = st.text_input("Notion 데이터베이스 ID를 입력하세요:")

# 사용자 질문 입력
user_question = st.text_input("질문을 입력하세요:")

if st.button("분석 시작"):
    if api_key and user_question:
        # OpenAI LLM 초기화
        llm = OpenAI(temperature=0)

        # 문서 처리 부분을 수정합니다
        with st.spinner("문서 처리 중..."):
            text_splitter = CharacterTextSplitter(chunk_size=3000, chunk_overlap=200)
            
            if source_type == "텍스트":
                texts = [Document(page_content=doc_content, metadata={})]
            elif source_type == "PDF" and uploaded_file:
                pdf_loader = PyPDFLoader(uploaded_file)
                texts = text_splitter.split_documents(pdf_loader.load())
            elif source_type == "YouTube" and youtube_url:
                youtube_loader = YoutubeLoader.from_youtube_url(youtube_url, add_video_info=True)
                texts = text_splitter.split_documents(youtube_loader.load())
            elif source_type == "Notion" and notion_db_id:
                notion_loader = NotionDBLoader(integration_token=notion_token, database_id=notion_db_id)
                texts = text_splitter.split_documents(notion_loader.load())
            else:
                st.error("필요한 정보를 모두 입력해주세요.")
                st.stop()

            # 디버그 출력 추가
            st.write("Texts 내용:", type(texts), len(texts))
            if texts:
                st.write("첫 번째 요소 타입:", type(texts[0]))

            # 메타데이터에서 복잡한 값 필터링 (수정된 부분)
            filtered_texts = []
            for doc in texts:
                if isinstance(doc, Document):
                    filtered_texts.append(filter_complex_metadata(doc))
                elif isinstance(doc, tuple) and len(doc) == 2:
                    filtered_texts.append(Document(page_content=doc[0], metadata=doc[1]))
                else:
                    st.warning(f"예상치 못한 형식의 데이터: {type(doc)}")
                    filtered_texts.append(doc)

            texts = filtered_texts

            # 임베딩 및 벡터 저장소 생성
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_documents(texts, embeddings)

            # RAG 시스 생성
            retriever = vectorstore.as_retriever()
            rag = RetrievalQA.from_chain_type(llm=llm, chain_type="stuff", retriever=retriever)

        # YouTube 검색 API 래퍼
        youtube_search = search()

        # 도구 정의
        tools = [
            Tool(
                name="RAG 시스템",
                func=rag.run,
                description="문서에서 정보를 검색하고 질문에 답변할 때 사용합니다."
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
        with st.spinner("에이전트가 분석 중입니다..."):
            response = agent.run(user_question)

        st.subheader("분석 결과:")
        st.write(response)
    else:
        st.error("API 키와 질문을 모 입력해주세요.")

# Streamlit 앱 실행 방법 안내
st.sidebar.markdown("""
## 앱 실행 방법
1. 필요한 API 키를 입력하세요.
2. 입력 소스를 선택하고 필요한 정보를 제공하세요.
3. 질문을 입력하세요.
4. "분석 시작" 버튼을 클릭하세요.
""")

# 주의사항
st.sidebar.warning("""
주의: 이 앱은 OpenAI, Notion, YouTube의 API를 사용합니다. 
각 서비스의 사용 제한과 비용을 확인하세요.
""")



