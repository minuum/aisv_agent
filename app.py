import streamlit as st
from langchain.agents import AgentType, initialize_agent
from langchain_community.llms import OpenAI
from langchain.tools import Tool
from langchain_community.embeddings import OpenAIEmbeddings
from langchain_community.vectorstores import Chroma
from langchain_community.document_loaders import TextLoader
from langchain.text_splitter import CharacterTextSplitter
from langchain.chains import RetrievalQA
import os

# Streamlit 앱 제목 설정
st.title("LangChain RAG 시스템 및 에이전트")

# OpenAI API 키 입력
api_key = st.text_input("OpenAI API 키를 입력하세요:", type="password")
os.environ["OPENAI_API_KEY"] = api_key

# 문서 내용 입력
doc_content = st.text_area("분석할 문서 내용을 입력하세요:")

# 사용자 질문 입력
user_question = st.text_input("질문을 입력하세요:")

if st.button("분석 시작"):
    if api_key and doc_content and user_question:
        # OpenAI LLM 초기화
        llm = OpenAI(temperature=0)

        # 문서 처리
        with st.spinner("문서 처리 중..."):
            text_splitter = CharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
            texts = text_splitter.split_text(doc_content)

            # 임베딩 및 벡터 저장소 생성
            embeddings = OpenAIEmbeddings()
            vectorstore = Chroma.from_texts(texts, embeddings)

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
        st.error("API 키, 문서 내용, 그리고 질문을 모두 입력해주세요.")