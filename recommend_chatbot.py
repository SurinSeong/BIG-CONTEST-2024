import streamlit as st 
import pandas as pd
from PIL import Image
##################################################################api
import google.generativeai as genai
##################################################################vectordb
from langchain_chroma import Chroma
##################################################################embedding
from langchain_huggingface import HuggingFaceEmbeddings
from defs import (clear_chat_history, main)

api_key = st.secrets['api_keys']["GEMINI_OPEN_API_KEY"]
genai.configure(api_key=api_key)

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
    
    st.subheader('지역을 선택하세요! 해당 지역의 맛집을 찾아드릴게요.')
    st.write("")
    
    # 체크박스 사용
    local_jeju_city = st.checkbox('제주시')  # 제주시 체크박스
    local_seogwipo_city = st.checkbox('서귀포시')  # 서귀포시 체크박스
    st.write("")
    
    # PNG 이미지 삽입 (제주도 지도.png 이미지 삽입!!!!!!!!!!!!)
    image = Image.open(r'C:\Users\tjdtn\inflearn-llm-application\big-contest-streamlit\제주도 지도.png')  # 이미지 파일 경로
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
 
for message in st.session_state.messages:  
    with st.chat_message(message["role"]):
        st.write(message["content"])
               
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당을 찾으시나요?"}]
st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

########################### 검색형 데이터 csv ########################### 
# 해당 데이터 경로로 변경 하세요!!   
path = r'C:\Users\tjdtn\inflearn-llm-application\big-contest-streamlit\JEJU_MCT_DATA_v2(12월)_v2.csv'
raw = pd.read_csv(path, index_col = 0)
df = raw.copy()

#########################임베딩 모델 로드##############################    
embedding_function = HuggingFaceEmbeddings(model_name='jhgan/ko-sroberta-multitask')

#############################ChromaDB##############################    
# ChromaDB 불러오기
# 해당 데이터 경로로 변경 하세요!!
recommendation_store = Chroma(
    collection_name='jeju_store_mct_keyword_6',
    embedding_function=embedding_function,
    persist_directory= r'C:\Users\tjdtn\inflearn-llm-application\big-contest-streamlit\chromadb\mct_keyword_v6'
)
# metadata 설정
metadata = recommendation_store.get(include=['metadatas'])

###########################################사용자 입력 쿼리################################################
# 사용자 입력에 따른 검색
if user_input := st.chat_input():
    st.session_state.messages.append({"role": "user", "content": user_input})
    with st.chat_message("user"):
        st.write(user_input)
        
    with st.spinner("음식점을 찾는 중입니다..."):    
        # 음식점 검색 및 결과 반환
        response = main(user_input, df)
        
        # 사용자가 제주시를 선택하고 서귀포시 음식점을 요청한 경우
        if (local_jeju_city) and (not local_seogwipo_city) and ('서귀포' in user_input):
            assistant_response = "제주시에 있는 음식점만 추천해드릴 수 있어요. 서귀포시에 있는 음식점을 추천받고 싶다면 서귀포시에 체크해주세요."
        
        # 사용자가 서귀포시를 선택하고 제주시에 있는 음식점을 요청한 경우
        elif (local_seogwipo_city) and (not local_jeju_city) and ('제주' in user_input):
            assistant_response = "서귀포시에 있는 음식점만 추천해드릴 수 있어요. 제주시에 있는 음식점을 추천받고 싶다면 제주시에 체크해주세요."

        # 검색 결과가 있는 경우
        elif response:
            assistant_response = response  # 검색 결과를 assistant_response로 저장
        
        # 검색 결과가 없을 때
        else:
            assistant_response = "질문해주신 음식점을 찾지 못했습니다. 다시 질문해주세요."

    # 챗봇 응답 메시지 추가
    st.session_state.messages.append({"role": "assistant", "content": assistant_response})

    # 챗봇 응답 출력
    with st.chat_message("assistant"):
        st.write(assistant_response)
        
print(st.session_state)