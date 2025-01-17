import streamlit as st 
import pandas as pd
import torch
import os
from PIL import Image
##################################################################api
from dotenv import load_dotenv
import google.generativeai as genai
##################################################################vectordb
import chromadb
from langchain_chroma import Chroma
##################################################################embedding
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings
from sentence_transformers import SentenceTransformer
##################################################################answer
from langchain.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_google_genai import GoogleGenerativeAI

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

##################################################################
#chatbot UI
st.set_page_config(page_title='제주도 맛집', page_icon="🏆",initial_sidebar_state="expanded")
st.title('제주도 음식점 탐방!')
st.subheader("누구랑 제주도 왔나요? 맞춤 제주도 맛집 추천해드려요~")

st.write("")

st.write("#연인#아이#친구#부모님#혼자#반려동물 #데이트#나들이#여행#일상#회식#기념일...")
st.write("")

with st.sidebar:
    st.title('<옵션을 선택하면 빠르게 추천해드려요!>')
    st.write("")
    
    st.subheader('지역을 선택하세요! 해당 지역의 맛집을 찾아드릴께요.')
    st.write("")
    
    # 체크박스 사용
    local_jeju_city = st.checkbox('제주시')  # 제주시 체크박스
    local_seogwipo_city = st.checkbox('서귀포시')  # 서귀포시 체크박스
    st.write("")
    
    # PNG 이미지 삽입
    image = Image.open(r'D:\2024_bigcontest\data\이미지\제주도 지도.png')  # 이미지 파일 경로
    st.image(image, caption='제주도 지도', use_column_width=True)  # 사이드바에 이미지 삽입

# Store LLM generated responses
if "messages" not in st.session_state:
    st.session_state.messages = []

# Check if the initial assistant message has been displayed
if "message_displayed" not in st.session_state:
    st.session_state.message_displayed = False

# Display the initial assistant message only once
if not st.session_state.message_displayed:
    st.session_state.messages.append({"role": "assistant", "content": "어떤 식당을 찾으시나요?"})
    st.session_state.message_displayed = True  # Mark message as displayed

# Display previous messages if they exist
# for message in st.session_state.messages:
#     with st.chat_message(message["role"]):
#         st.write(message["content"])
        
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당을 찾으시나요?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)
  
##################################################################    
#llm 함수 
def get_llm():
    generation_config = {
        "temperature": 0,
        "top_p": 0.95,
        "top_k": 64,
        "max_output_tokens": 8192,
        "response_mime_type": "text/plain",
    }

    # GoogleGenerativeAI 모델 초기화
    llm = GoogleGenerativeAI(
        model="gemini-1.5-flash",   # 모델 이름 설정
        api_key=gemini_api_key,  # 필수 입력 필드
        **generation_config          # 추가 설정값 전달
    )
    return llm

#########################임베딩 모델 로드##############################    
@st.cache_resource
def load_embedding_model():
    model_name = "jhgan/ko-sroberta-multitask"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    embedding_function = HuggingFaceEmbeddings(model_name=model_name)
    return tokenizer, model, embedding_function

tokenizer, model, embedding_function = load_embedding_model()

#########################임베딩 함수##############################    
# 텍스트 임베딩
def embed_text(text):
    inputs = tokenizer(text, return_tensors='pt', padding=True, truncation=True)
    with torch.no_grad():
        embeddings = model(**inputs).last_hidden_state.mean(dim=1)
    return embeddings.squeeze().cpu().numpy()

###########################ChromaDB#############################
# ChromaDB 데이터 저장 경로 설정 (이전에 저장했던 경로)
embedding_function = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

# ChromaDB 불러오기
db_folder = r'D:\2024_bigcontest\VectorDB\mct_keyword_v5'
client = chromadb.PersistentClient(path=db_folder)
collection = client.get_collection("jeju_store_mct_keyword_5")

# search_retriever = search_store.as_retriever(search_kwargs={'k' : 10})
###########################retreiver##########################
# 음식점 검색 함수 정의
def search_restaurants(query):
    query_embedding = embed_text(query)  # 질문 임베딩
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=5  # 반환할 결과 수
    )
    # '제주시' 주소가 포함된 메타데이터 필터링
    filtered_results = []
    if local_jeju_city:
        for metadata in results['metadatas'][0]:
            if '제주시' in metadata.get('address', ''):  # 'address'에 '제주시'가 포함된 데이터 
                filtered_results.append(metadata)
    elif local_seogwipo_city:
        for metadata in results['metadatas'][0]:
            if '서귀포시' in metadata.get('address', ''):  # 'address'에 '서귀포시'가 포함된 데이터 
                filtered_results.append(metadata)
    else:
        for metadata in results['metadatas'][0]: # 제주시 & 서귀포시가 포함된 데이터 
            filtered_results.append(metadata)

    return filtered_results

##################################################################
# 응답을 생성하는 함수
def generate_response(user_input):
    # 각 문서에서 필요한 정보를 추출하여 요약문 생성
    search_results = search_restaurants(user_input) 
       
    # 검색 결과가 없을 때 처리
    if not search_results:  # 결과가 빈 리스트일 경우
        return "물어보신 음식점은 존재하지 않습니다."
     
    responses = []
    for doc in search_results:  # 상위 3개의 문서만 사용
        context = {
        "name": doc['name'], #가게명
        "address": doc['address'], #주소
        "industry": doc['industry'], #업종
        "attraction": ', '.join(doc['attraction'][:3]), #주변 관광지
        "summary": doc['summary'] # 요약 
    } 
    
# LLM에게 주어진 문맥을 바탕으로 응답을 생성하도록 요청
    prompt_template  = """
    다음의 가게 정보를 바탕으로 각 가게에 대한 추천 내용을 만들어 주세요:
    가게 정보 :
    name: {name}
    address: {address}
    industry: {industry}
    attraction: {attraction}
    summary: {summary}
    ---
    다음의 질문을 바탕으로 해당되는 결과물(가게 정보) 반드시 다음 [예시]과 같이 출력이 되어야 합니다.
    ---
    [예시]
    가게 : 서은이네생고기 \n
    주소 : 제주 제주시 연동 274-30번지 1층 \n
    업종 : 육류,고기요리 \n
    주변 관광지 : 엠버호텔, 수목원테마파크, 롯데시티호텔 제주 \n
    가게 특징 : (질문에 대한 요약으로 보여주기) \n
    ---
    
    요약의 경우, query에 대한 요약으로 보여주세요.
    만일, 위의 형태로 답변이 나오지 않는 결과물은 메시지를 보여주지 마세요.
    불필요한 기호는 없애고 작성해주세요.
    """
    
    # 프롬프트에 데이터 채우기
    prompt = prompt_template.format(**context)
    
    # LLM을 이용해 응답 생성
    llm = get_llm()  # 사용 중인 LLM을 가져오는 함수
    response = llm(prompt)
    responses.append(response)

    # 모든 제품에 대한 추천 문장을 결합하여 최종 응답 생성
    final_response = response
        
    return final_response
##################################################################################################
# 사용자 입력에 따른 검색
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    # 음식점 검색 및 결과 반환
    response = generate_response(user_input)
        
    # 챗봇 응답 메시지 추가
    st.session_state.messages.append({"role": "assistant", "content": response})

    # 챗봇 응답 출력
    with st.chat_message("assistant"):
        st.write(response)

# 이전 메시지 출력 (세션 상태 유지)
for message in st.session_state.messages:  
    with st.chat_message(message["role"]):
        st.write(message["content"])    
    
