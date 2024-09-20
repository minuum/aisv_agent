import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate
from langchain.schema import StrOutputParser
from langgraph.graph import StateGraph, END
from typing import Dict, TypedDict, Annotated
import os
from dotenv import load_dotenv
from langchain_openai import OpenAIEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain.chains import RetrievalQA
import time
from langchain_community.tools import DuckDuckGoSearchRun
import requests
from bs4 import BeautifulSoup
from urllib.parse import urlparse
import re
from langchain.prompts import PromptTemplate
from langchain_community.utilities import SerpAPIWrapper
import queue

# 디버그 로그를 저장할 전역 큐
debug_log_queue = queue.Queue()

# 환경 변수 로드
load_dotenv()

# OpenAI API 키 설정
os.environ["OPENAI_API_KEY"] = os.getenv("OPENAI_API_KEY")

# FAISS 인덱스 경로 설정
FAISS_INDEX_PATH = "faiss_index"

# PDF 파일 로드 및 벡터 저장소 생성
@st.cache_resource
def create_vector_store():
    if os.path.exists(FAISS_INDEX_PATH):
        st.write("기존 FAISS 인덱스를 로드합니다.")
        vector_store = FAISS.load_local(FAISS_INDEX_PATH, OpenAIEmbeddings(), allow_dangerous_deserialization=True)
    else:
        st.write("FAISS 인덱스가 없습니다. 새로 생성합니다.")
        vector_store = create_new_vector_store()
    
    return vector_store

def create_new_vector_store():
    st.write("새로운 벡터 저장소를 생성합니다.")
    loader = PyPDFLoader("2023_상반기_법령해석사례집(상).pdf")
    documents = loader.load()
    
    text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=200)
    texts = text_splitter.split_documents(documents)
    
    embeddings = OpenAIEmbeddings()
    vector_store = FAISS.from_documents(texts, embeddings)
    
    # 벡터 저장소를 로컬에 저장
    vector_store.save_local(FAISS_INDEX_PATH)
    
    return vector_store

# 벡터 저장소 생성 또는 로드
vector_store = create_vector_store()

# 상태 정의
class State(TypedDict):
    question: str
    legal_area: str
    search_results: str
    answer: str
    feedback: str
    score: int

# RAG 체인 생성
llm = ChatOpenAI(model="gpt-4o-mini-2024-07-18", temperature=0) # Don't change the model
rag_chain = RetrievalQA.from_chain_type(
    llm=llm,
    chain_type="stuff",
    retriever=vector_store.as_retriever(),
    return_source_documents=True
)

# 검색 도구 초기화
serpapi_api_key = os.getenv("SERPAPI_API_KEY")  # SerpAPI 키를 환경 변수에서 가져옵니다
search = SerpAPIWrapper(serpapi_api_key=serpapi_api_key)

# 노드 함수 정의
def question_analysis(state: State) -> Dict:
    log_debug("질문 분석 시작", 1)
    prompt = ChatPromptTemplate.from_template(
        "사용자의 질문을 분석하고 법률 분야를 판단해주세요. 질문: {question}"
    )
    chain = prompt | llm | StrOutputParser()
    legal_area = chain.invoke({"question": state["question"]})
    log_debug(f"분석된 법률 분야: {legal_area}", 1)
    return {"legal_area": legal_area}

def search_information(state: State) -> Dict:
    log_debug("정보 검색 시작", 2)
    query = f"{state['legal_area']} {state['question']}"
    rag_results = rag_chain.invoke(query)
    
    log_debug("인터넷 검색 시작", 2)
    try:
        search_results = search.run(query)
        log_debug(f"인터넷 검색 결과: {search_results[:500]}...", 2)  # 결과의 일부만 로그에 기록
    except Exception as e:
        log_debug(f"인터넷 검색 중 오류 발생: {str(e)}", 2)
        search_results = "인터넷 검색 실패 또는 결과 없음"
    
    log_debug("RAG 및 인터넷 검색 완료", 2)
    return {
        "rag_results": rag_results['result'] if isinstance(rag_results, dict) and 'result' in rag_results else str(rag_results),
        "search_results": search_results
    }

def combine_information(state: State) -> Dict:
    log_debug("정보 결합 시작", 3)
    combined_info = f"RAG 결과: {state.get('rag_results', '결과 없음')}\n\n검색 결과: {state.get('search_results', '결과 없음')}"
    log_debug("정보 결합 완료", 3)
    return {
        "combined_info": combined_info,
        "question": state["question"],
        "legal_area": state.get("legal_area", ""),
        "search_results": state.get("search_results", ""),
        "rag_results": state.get("rag_results", "")
    }

def generate_answer(state: State) -> Dict:
    log_debug("답변 생성 시작", 4)
    prompt_template = PromptTemplate(
        input_variables=["question", "combined_info"],
        template="""
        당신은 법률 전문 AI 상담사입니다. 주어진 정보를 바탕으로 사용자의 법률 관련 질문에 대해 명확하고 정확한 답변을 제공해야 합니다.

        질문: {question}

        관련 정보:
        {combined_info}

        위의 정보를 바탕으로 다음 형식에 맞춰 답변을 작성해 주세요:

        1. 법률 분야: [질문과 관련된 법률 분야를 간단히 명시]

        2. 답변 요약: [질문에 대한 핵심 답변을 1-2문장으로 요약]

        3. 상세 설명:
        [관련 법률 조항이나 판례를 인용하며 더 자세한 설명 제공]

        4. 주의사항:
        [해당 사안에 대한 주의점이나 추가로 고려해야 할 사항 언급]

        5. 추천 행동:
        [질문자가 취해야 할 다음 단계나 행동 제안]

        6. 면책 조항:
        이 답변은 일반적인 법률 정보 제공 목적으로 작성되었으며, 구체적인 법률 자문을 대체할 수 없습니다. 정확한 법률 자문을 위해서는 반드시 변호사와 상담하시기 바랍니다.
        """
    )
    
    chain = prompt_template | llm | StrOutputParser()
    
    try:
        combined_info = state.get("combined_info", "정보 없음")
        answer = chain.invoke({"question": state["question"], "combined_info": combined_info})
    except Exception as e:
        log_debug(f"답변 생성 중 오류 발생: {str(e)}", 4)
        answer = "죄송합니다. 답변을 생성하는 중에 오류가 발생했습니다."
    
    log_debug("답변 생성 완료", 4)
    return {"answer": answer}

def evaluate_answer(state: State) -> Dict:
    log_debug("답변 평가 시작", 5)
    prompt = ChatPromptTemplate.from_template(
        """
        다음 답변의 품질을 1부터 10까지의 점수로 평가해주세요. 
        점수만 숫자로 반환해 주세요.
        
        질문: {question}
        답변: {answer}
        """
    )
    chain = prompt | llm | StrOutputParser()
    evaluation = chain.invoke({"question": state["question"], "answer": state["answer"]})
    try:
        score = int(evaluation.strip())
        result = "END" if score >= 8 else "generate_answer"
    except ValueError:
        log_debug(f"점수 변환 실패. 원본 평가: {evaluation}", 5)
        score = 0
        result = "generate_answer"
    log_debug(f"평가 결과: {result} (점수: {score})", 5)
    return {"feedback": str(score), "result": result, "score": score}

# 그래프 정의
workflow = StateGraph(State)

workflow.add_node("question_analysis", question_analysis)
workflow.add_node("search_information", search_information)
workflow.add_node("combine_information", combine_information)
workflow.add_node("generate_answer", generate_answer)
workflow.add_node("evaluate_answer", evaluate_answer)

workflow.set_entry_point("question_analysis")

# 엣지 정의
workflow.add_edge("question_analysis", "search_information")
workflow.add_edge("search_information", "combine_information")
workflow.add_edge("combine_information", "generate_answer")
workflow.add_edge("generate_answer", "evaluate_answer")

# 조건부 엣지 추가
workflow.add_conditional_edges(
    "evaluate_answer",
    lambda x: x["result"],
    {
        "generate_answer": "generate_answer",
        "END": END
    }
)

# 그래프 컴파일
app = workflow.compile()

# 앱 인터페이스
st.title("법률 AI 상담 시스템")

# 모드 선택
mode = st.radio("모드 선택", ["RAG", "RAG Langgraph"])

# 디버그 모드 토글 추가
debug_mode = st.sidebar.checkbox("디버그 모드", value=True)

# 디버그 로그를 위한 컨테이너 생성
debug_container = st.sidebar.container()

# 시간 측정을 위한 컨테이너 생성
timer_container = st.sidebar.empty()

# 로그 출력 함수
def log_debug(message, step=None):
    if step:
        message = f"Step {step}: {message}"
    debug_log_queue.put(message)
    print(message)  # 터미널에도 출력

# URL 추출 함수
def extract_urls(text):
    url_pattern = re.compile(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\\(\\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+')
    return url_pattern.findall(text)

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []
if "debug_logs" not in st.session_state:
    st.session_state.debug_logs = []

# 이전 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력
if prompt := st.chat_input("법률 관련 질문을 입력해주세요:"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # AI 응답 생성
    with st.chat_message("assistant"):
        full_response = ""
        message_placeholder = st.empty()


        if mode == "RAG":
            # RAG 모드: FAISS 인덱스 테스트
            try:
                log_debug("RAG 모드 시작", 0)
                start_time = time.time()
                
                log_debug("RAG 체인 실행", 1)
                response = rag_chain.invoke(prompt)
                log_debug("인터넷 검색 실행", 2)
                search_results = search.run(prompt)
                
                full_response = ""
                message_placeholder = st.empty()
                full_response = ""
                
                log_debug("답변 스트리밍 시작", 3)
                for chunk in response['result'].split():
                    full_response += chunk + " "
                    if chunk.endswith(('.', '!', '?', '\n')):
                        full_response += "\n\n"
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")
                
                message_placeholder.markdown(full_response)
                
                log_debug("출처 표시", 4)
                st.markdown("### 출처")
                cols = st.columns(3)
                for i, doc in enumerate(response['source_documents'], 1):
                    source = doc.metadata.get('source', 'Unknown')
                    page = doc.metadata.get('page', 'N/A')
                    content = doc.page_content[:100] + "..." if len(doc.page_content) > 100 else doc.page_content
                    
                    with cols[i % 3]:
                        with st.expander(f"출처 {i}: {source} (페이지 {page})"):
                            st.markdown(content)
                
                log_debug("인터넷 검색 결과 표시", 5)
                st.markdown("### 인터넷 검색 결과")
                cols = st.columns(3)
                urls = extract_urls(search_results)
                for i, url in enumerate(urls):
                    title, image_url = get_webpage_info(url)
                    with cols[i % 3]:
                        with st.expander(title):
                            st.markdown(f"[{url}]({url})")
                            if image_url:
                                st.image(image_url, width=200)
                
                elapsed_time = time.time() - start_time
                timer_container.markdown(f"⏱️ 경과 시간: {elapsed_time:.2f}초")
                
                log_debug("RAG 모드 종료", 6)
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
                log_debug(f"오류 상세 정보: {type(e).__name__}, {str(e)}")
                import traceback
                log_debug(f"스택 트레이스: {traceback.format_exc()}")
        
        elif mode == "RAG Langgraph":
            # RAG Langgraph 모드
            initial_state = State(
                question=prompt,
                legal_area="",
                rag_results="",
                search_results="",
                combined_info="",
                answer="",
                feedback="",
                score=0  # 초기 점수 추가
            )
             
            try:
                log_debug("RAG Langgraph 모드 시작", 0)
                start_time = time.time()
                max_iterations = 10  # 최대 반복 횟수 설정
                full_response = ""
                message_placeholder = st.empty()
                
                for step, output in enumerate(app.stream(initial_state), 1):
                    if step > max_iterations:
                        log_debug(f"최대 반복 횟수 ({max_iterations})에 도달하여 종료", step)
                        break
                    
                    log_debug(f"Step {step} 실행", step)
                    if isinstance(output, Dict):
                        for key, value in output.items():
                            if key in ["legal_area", "rag_results", "search_results", "combined_info", "answer", "feedback", "score"]:
                                if key == "score":
                                    full_response += f"현재 답변 점수: {value}\n\n"
                                elif key == "answer":
                                    # 답변을 단어 단위로 스트리밍
                                    current_answer = ""
                                    for word in value.split():
                                        current_answer += word + " "
                                        full_response = f"{full_response}{current_answer}▌"
                                        message_placeholder.markdown(full_response)
                                        time.sleep(0.05)
                                    full_response = full_response[:-1] + "\n\n"  # 마지막 '▌' 제거 및 줄바꿈 추가
                                else:
                                    full_response += f"{key.capitalize()}:\n{value}\n\n"
                                
                                message_placeholder.markdown(full_response)
                                
                                log_message = f"{key}: {value[:500]}..." if isinstance(value, str) and len(value) > 500 else f"{key}: {value}"
                                log_debug(log_message, step)
                                st.session_state.debug_logs.append(log_message)
                            
                        if key == "search_results":
                            log_debug("인터넷 검색 결과 처리", step)
                            st.markdown("### 인터넷 검색 결과")
                            urls = extract_urls(value)
                            if urls:
                                cols = st.columns(3)
                                for i, url in enumerate(urls):
                                    title, image_url = get_webpage_info(url)
                                    with cols[i % 3]:
                                        with st.expander(title):
                                            st.markdown(f"[{url}]({url})")
                                            if image_url:
                                                st.image(image_url, width=200)
                            else:
                                st.warning("검색 결과에서 URL을 찾을 수 없습니다.")
                                log_debug("검색 결과에서 URL을 찾을 수 없음", step)
                    
                    elapsed_time = time.time() - start_time
                    timer_container.markdown(f"⏱️ 경과 시간: {elapsed_time:.2f}초")
                       
                    if "result" in output and output["result"] == "END":
                        log_debug(f"RAG Langgraph 모드 종료 (최종 점수: {output.get('score', 'N/A')})", step + 1)
                        # 최종 답변 출력
                        st.markdown("### 최종 답변")
                        st.markdown(output.get("answer", "답변을 생성하지 못했습니다."))
                        break
                    
            except Exception as e:
                st.error(f"오류 발생: {str(e)}")
                log_debug(f"오류 상세 정보: {type(e).__name__}, {str(e)}")
                import traceback
                log_debug(f"스택 트레이스: {traceback.format_exc()}")
            
            # 최종 답변이 출력되지 않았을 경우를 대비한 추가 출력
            if "answer" not in locals():
                st.warning("답변 생성 과정에서 문제가 발생했습니다. 다시 시도해 주세요.")
            elif not st.session_state.messages[-1]["content"]:
                st.markdown("### 최종 답변")
                st.markdown(locals().get("answer", "답변을 생성하지 못했습니다."))
            
            full_response = ""
            message_placeholder = st.empty()

            # 마지막 메시지의 내용을 가져옵니다
            if st.session_state.messages and isinstance(st.session_state.messages[-1], dict):
                last_message = st.session_state.messages[-1].get('content', '')
                for chunk in last_message.split():
                    full_response += chunk + " "
                    if chunk.endswith(('.', '!', '?', '\n')):
                        full_response += "\n\n"
                    time.sleep(0.05)
                    message_placeholder.markdown(full_response + "▌")

                message_placeholder.markdown(full_response)

                # 마지막 메시지를 업데이트합니다
                st.session_state.messages[-1]['content'] = full_response
            else:
                st.warning("메시지가 없거나 형식이 올바르지 않습니다.")

        # 디버그 로그 표시
        if debug_mode:
            debug_logs = []
            while not debug_log_queue.empty():
                debug_logs.append(debug_log_queue.get())
            debug_container.text_area("디버그 로그", "\n".join(debug_logs), height=300)

# 피드백 수집 (사이드바로 이동)
with st.sidebar:
    st.subheader("피드백")
    feedback = st.text_area("답변에 대한 피드백을 남겨주세요:")
    if st.button("피드백 제출"):
        if feedback:
            st.success("피드백이 제출되었습니다. 사합니다!")
            log_debug(f"피드백 제출: {feedback}")
        else:
            st.warning("피드백을 입력해주세요.")

def get_webpage_info(url):
    try:
        response = requests.get(url)
        soup = BeautifulSoup(response.text, 'html.parser')
        title = soup.title.string if soup.title else "제목 없음"
        og_image = soup.find("meta", property="og:image")
        image_url = og_image["content"] if og_image else None
        return title, image_url
    except:
        return "제목 없음", None