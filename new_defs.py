import os
import random
import streamlit as st

## 제미나이 로드 ##
import google.generativeai as genai

## api 키 불러오기 ##
from dotenv import load_dotenv

load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

## 임베딩 모델 ##
from langchain_huggingface import HuggingFaceEmbeddings

embedding_function = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

## ChromaDB 가져오기 ##
from langchain_chroma import Chroma

recommendation_store = Chroma(
    collection_name='jeju_store_mct_keyword_6',
    embedding_function=embedding_function,
    persist_directory= r'C:\Users\tjdtn\inflearn-llm-application\big-contest-streamlit\chromadb\mct_keyword_v6'
)

# metadata 설정하기
metadata = recommendation_store.get(include=['metadatas'])

## 닉네임 생성하기 ##
def generate_random_id():
    # 닉네임 (id) 생성
    jeju_nicknames = [
        "한라산바람", "오름여행자", "바다향기", "감귤연인", "돌하르방친구", "푸른섬나그네", 
        "섭지코지연인", "해녀이야기", "제주바다빛", "감성제주러", "한치도사", "제주하늘", 
        "돌담길여행자", "조랑말의꿈", "바람의섬", "우도탐험가", "평화의바다", "제주푸름", 
        "오름의숨결", "비양도의꿈", "올레길여행자", "새별오름러버", "제주향기", "애월바다러버", 
        "성산일출연인", "한라봉나그네", "비자림의추억", "해안도로러버", "구좌바다바람", "용눈이오름"
    ]
    return random.choice(jeju_nicknames)
    
def making_id():
    created_id = generate_random_id()
    st.session_state['session_id'] = created_id
    
## 모델 가져오기 ##
# 데이터 캐싱 이용
@st.cache_resource
def load_model():
    model = genai.GenerativeModel('gemini-1.5-flash',
                                  generation_config={"temperature": 0,
                                                     "max_output_tokens": 5000})
    print('model loaded...')
    return model

model = load_model()

## 초기화 함수 ##
def reset_session_state():
    st.session_state['messages'] = []
    st.session_state['session_id'] = ""  # ID 초기화
    st.session_state['is_logged_in'] = False
    st.session_state['chat_history'] = model.start_chat(history=[])
 
## 세션 기록 불러오기 ##
from langchain_core.chat_history import BaseChatMessageHistory
from langchain_community.chat_message_histories import ChatMessageHistory
    
def get_session_history(session_id: str) -> BaseChatMessageHistory:
    if session_id not in st.session_state['store']:  # 세션 ID가 store에 없는 경우
        # 새로운 ChatMessageHistory 객체를 생성하여 store에 저장
        st.session_state['store'][session_id] = ChatMessageHistory()
    return st.session_state['store'][session_id]  # 해당 세션 ID에 대한 세션 기록 반환
 
# 새로고침하기 전에 'messages'에 있는 내용 보여주기
def print_messages():    
    if 'chat_history' in st.session_state and len(st.session_state['chat_history']) > 0:
        # 'user'가 입력한 내용은 'user' 아이콘과 함께 나가야 하고, 'assistant'가 작성한 내용은 'assistant' 아이콘과 함께 나가야 한다.
        for content in st.session_state['chat_session'].history:
            with st.chat_message('ai' if content.role == 'model' else 'user'):
                st.markdown(content.parts[0].text)


 