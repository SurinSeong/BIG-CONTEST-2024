import os
from PIL import Image
import streamlit as st

from new_defs import (load_model, making_id, reset_session_state, get_session_history, print_messages)

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
# metadata 설정
metadata = recommendation_store.get(include=['metadatas'])

## 모델 불러오기 ##
model = load_model()

## chatbot UI ##
st.set_page_config(page_title='제주도 맛집', page_icon="🏆",initial_sidebar_state="expanded")

st.title('제주도 음식점 탐방!')
st.subheader("누구와 제주도에 오셨나요? 제주도 맛집 추천해드려요~")

st.write("")

st.write("#연인#아이#친구#부모님#혼자#반려동물 #데이트#나들이#여행#일상#회식#기념일...")
st.write("")

## session_id에 따른 메시지 히스토리 초기화 ##
# 모든 사용자의 대회기록을 저장하는 store 세션 상태 변수
if 'store' not in st.session_state:
    st.session_state['store'] = dict()

# 초기 메시지 시작할 때에 message container 만들어 이곳에 앞으로 저장
if 'messages' not in st.session_state:
    st.session_state['messages'] = [] # 아예 내용을 지우고 싶다면 리스트 안의 내용을 clear 해주면 된다.

# 채팅 대화기록을 저장하는 chat_session 세션 상태 변수
if 'chat_history' not in st.session_state:
    st.session_state['chat_history'] = model.start_chat(history=[]) # ChatSession 반환하기

# 로그인 여부
if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

# 세션 아이디
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = ""

## 사이드바 설정 ##
with st.sidebar:
    col1, col2 = st.columns(2)
    with col1:
        if st.button('닉네임 생성'):
            making_id()
        
    with col2:
        if st.button('로그아웃'):
            reset_session_state()
            st.rerun()
            
    session_id = st.text_input('Session ID', value=st.session_state['session_id'], key='session_id')
     
    if st.button('로그인'):
        if st.session_state.session_id:
            st.session_state['is_logged_in'] = True  # 로그인 상태 저장
            get_session_history(session_id)
        else:
            st.write("Session ID를 입력하세요.")
            
        # 로그인 성공 후 세션 상태 확인
        if st.session_state.get('is_logged_in', False):
            st.sidebar.write(f"현재 로그인된 닉네임: {st.session_state['session_id']}")
        else:
            st.sidebar.write("로그인 필요")
                
    ## 도시 선택 ##               
    st.title('<옵션을 선택하면 빠르게 추천해드려요!>')
    st.write("")
    
    st.subheader('지역을 선택하세요! 해당 지역의 맛집을 찾아드릴께요.')
    st.write("")
    
    # 체크박스 사용
    local_jeju_city = st.checkbox('제주시')  # 제주시 체크박스
    local_seogwipo_city = st.checkbox('서귀포시')  # 서귀포시 체크박스
    st.write("")
    
    # PNG 이미지 삽입
    image = Image.open(r'C:\Users\tjdtn\inflearn-llm-application\big-contest-streamlit\제주도 지도.png')  # 이미지 파일 경로
    st.image(image, caption='제주도 지도', use_column_width=True)  # 사이드바에 이미지 삽입

## 이전 대화기록을 출력 ##
print_messages()
get_session_history(session_id)


## 사용자 입력에 따른 검색 답변 주기 ##
if 'is_logged_in' not in st.session_state or not st.session_state['is_logged_in']:
        # ID가 없을 때 안내 메시지
        with st.chat_message('ai'):
            st.markdown('닉네임을 생성하고 로그인해주세요.')
      
else:
    if user_input := st.chat_input('반갑습니다. 어떤 음식점을 찾고 계신가요?'):
        # 사용자가 입력한 내용을 출력
        with st.chat_message('user'):
            st.markdown(f'{user_input}') 
        
        with st.spinner("음식점을 찾는 중입니다..."):
            # AI의 답변을 출력
            with st.chat_message('ai'):
                message_placeholder = st.empty() # DeltaGenerator 반환
                full_response = ""
                
                # 제주 선택했는데 서귀포 물어보면
                if (local_jeju_city) and (not local_seogwipo_city) and ('서귀포' in user_input):
                    response = '제주시에 있는 음식점만 추천해드릴 수 있어요.  서귀포시에 있는 음식점을 추천받고 싶다면 서귀포시에 체크해주세요.'
                    message_placeholder.markdown(response)
                
                # 서귀포 선택했는데 제주 물어보면    
                elif (local_seogwipo_city) and (not local_jeju_city) and ('제주' in user_input):
                    response = "서귀포시에 있는 음식점만 추천해드릴 수 있어요. 제주시에 있는 음식점을 추천받고 싶다면 제주시에 체크해주세요."
                    message_placeholder.markdown(response)
                
                else:
                    response = st.session_state.chat_session.send_message(user_input, stream=True)
                    for chunk in response:
                        full_response += chunk.text
                        message_placeholder.markdown(full_response)

#----------------------------------------------------------------
            
        with st.spinner("음식점을 찾는 중입니다..."):
            # AI의 답변 생성
            with st.chat_message('ai'):
                # 우선 옳지 않은 질문은 다시 질문하라고 하기
                # 사용자가 제주시를 선택하고 서귀포시 음식점을 요청한 경우
                if (local_jeju_city) and (not local_seogwipo_city) and ('서귀포' in user_input):
                    assistant_response = "제주시에 있는 음식점만 추천해드릴 수 있어요. 서귀포시에 있는 음식점을 추천받고 싶다면 서귀포시에 체크해주세요."
                    # 답변 저장하기
                    st.session_state['messages'].append(ChatMessage(role='ai', message=assistant_response))
                
                # 사용자가 서귀포시를 선택하고 제주시에 있는 음식점을 요청한 경우
                elif (local_seogwipo_city) and (not local_jeju_city) and ('제주' in user_input):
                    assistant_response = "서귀포시에 있는 음식점만 추천해드릴 수 있어요. 제주시에 있는 음식점을 추천받고 싶다면 제주시에 체크해주세요."
                    # 답변 저장하기
                    st.session_state['messages'].append(ChatMessage(role='ai', message=assistant_response)) 

                # 장소에 맞게 질문이 잘 들어 온 경우
                else:
                    main_runnable = RunnableLambda(
                        lambda inputs: main(inputs['question'], inputs['session_id'])
                    )
                    
                    # 실제 Runnable
                    with_message_history = RunnableWithMessageHistory(
                        main_runnable,
                        get_session_history,
                        input_messages_key='question',
                        history_messages_key='history'
                    )
                    
                    # 질문이 들어오면 실행하기
                    response = with_message_history.invoke(
                        {'question':user_input, 'session_id':st.session_state['session_id']},
                        config={'configurable': {'session_id':st.session_state['session_id']}}
                    )
                    
                    # 최종으로 invoke한 내용을 response에 넣고 contents에 저장
                    st.session_state['message'].append(ChatMessage(role='ai', message=response))
            
print(st.session_state)