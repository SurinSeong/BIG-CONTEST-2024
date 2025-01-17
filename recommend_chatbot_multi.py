import streamlit as st 
import pandas as pd
import os
from PIL import Image
##################################################################api
from dotenv import load_dotenv
import google.generativeai as genai
##################################################################vectordb
from langchain_chroma import Chroma
##################################################################embedding
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.runnables import(
    RunnableLambda
)
from langchain_core.messages import ChatMessage
from langchain_core.runnables.history import RunnableWithMessageHistory
from defs_multi import (clear_chat_history, main, reset_session_state, making_id,
                  get_session_history, category_classification, print_messages)

#.env 파일 생성해서 GEMINI_API_KEY=API_KEY 입력 후 실행하시면돼요
load_dotenv()

gemini_api_key = os.getenv('GEMINI_API_KEY')
genai.configure(api_key=gemini_api_key)

##################################################################
#chatbot UI
st.set_page_config(page_title='제주도 맛집', page_icon="🏆",initial_sidebar_state="expanded")
st.title('제주도 음식점 탐방!')
st.subheader("누구와 제주도에 오셨나요? 제주도 맛집 추천해드려요~")

st.write("")

st.write("#연인#아이#친구#부모님#혼자#반려동물 #데이트#나들이#여행#일상#회식#기념일...")
st.write("")


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

########################메시지 초기화 구간#####################

# 초기 메시지 시작할 때에 message container 만들어 이곳에 앞으로 저장
if 'messages' not in st.session_state:
    st.session_state['messages'] = [] # 아예 내용을 지우고 싶다면 리스트 안의 내용을 clear 해주면 된다.

# 채팅 대화기록을 저장하는 store 세션 상태 변수
if 'store' not in st.session_state:
    st.session_state['store'] = dict()

# 로그인 여부
if 'is_logged_in' not in st.session_state:
    st.session_state['is_logged_in'] = False

# 세션 아이디
if 'session_id' not in st.session_state:
    st.session_state['session_id'] = ""

# if "message_displayed" not in st.session_state:
#     st.session_state.message_displayed = False

## 사이드바 ## 
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
    
    # PNG 이미지 삽입 (제주도 지도.png 이미지 삽입!!!!!!!!!!!!)
    image = Image.open(r'C:\Users\tjdtn\inflearn-llm-application\big-contest-streamlit\제주도 지도.png')  # 이미지 파일 경로
    st.image(image, caption='제주도 지도', use_column_width=True)  # 사이드바에 이미지 삽입



# ## 메시지 입력창에..
# if 'is_logged_in' not in st.session_state or not st.session_state['is_logged_in']:
#     # ID가 없으면 안내 메시지 준다
#     st.chat_message("assistant").write("어떤 식당을 찾으세요? ID를 받아 로그인해주시고 질문해주세요.") # 멘트 변경 필요
# else:
#     # ID 있으면
#     if user_input := st.chat_input('어떤 음식점을 찾고 있으세요?'):
#         # 사용자가 입력한 내용을 출력
#         st.chat_message('user').write(f'{user_input}')
#         st.session_state['message'].append(ChatMessage(role='user', content=user_input))
        
#         with st.spinner("음식점을 찾는 중입니다.."):
#             # AI 답변 생성
#             with st.chat_message('assistant'):
#                 process_user_query_runnable = RunnableLambda(
#                     lambda inputs: p
#                 )
    

               
# st.sidebar.button('Clear Chat History', on_click=clear_chat_history)


############################# 이전 대화기록 누적/출력해 주는 코드 #############################
# 이전 대화기로기을 출력해 주는 코드
print_messages()
get_session_history(session_id) 

###########################################사용자 입력 쿼리################################################
# 사용자 입력에 따른 검색
if 'is_logged_in' not in st.session_state or not st.session_state['is_logged_in']:
        # ID가 없을 때 안내 메시지
        st.chat_message('ai').write("닉네임을 생성하고 로그인해주세요.")
else:
    if user_input := st.chat_input('반갑습니다. 어떤 음식점을 찾고 계신가요?'):
        # 사용자가 입력한 내용을 출력
        st.chat_message('user').write(f'{user_input}') 
        st.session_state['messages'].append(ChatMessage(role='user', message=user_input, content=user_input))
        
            
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
        #     # 음식점 검색 및 결과 반환
        #     response = main(user_input, df, session_id)
            
            

        #     # 검색 결과가 있는 경우
        #     elif response:
        #         assistant_response = response  # 검색 결과를 assistant_response로 저장
            
        #     # 검색 결과가 없을 때
        #     else:
        #         assistant_response = "질문해주신 음식점을 찾지 못했습니다. 다시 질문해주세요."

        # # 챗봇 응답 메시지 추가
        # st.session_state.messages.append({"role": "assistant", "content": assistant_response})
        
        # # 챗봇 응답 출력
        # with st.chat_message("assistant"):
        #     # 세션 기록을 기반으로 한 RunnableWithMessageHistory 설정
        #                 process_user_query_runnable = RunnableLambda(
        #                     lambda inputs: main(inputs["question"], df, inputs["session_id"])
        #                 )

        #                 # 실제 RunnableWithMessageHistory 가 적용된 Chain
        #                 with_message_history = RunnableWithMessageHistory(
        #                     process_user_query_runnable,
        #                     get_session_history,
        #                     input_messages_key="question",
        #                     history_messages_key="history",
        #                 )
                        
        #                 # 질문이 들어오면 실행 (chain 실행)
        #                 response = with_message_history.invoke(
        #                     {"question": user_input, "session_id": st.session_state['session_id']},  
        #                     config={"configurable": {"session_id": st.session_state['session_id']}}
        #                 )
                    
        #                 # 최종 invoke한 내용을 response에 넣었고 그것을 contents에 저장
        #                 st.session_state.messages.append({"role": "assistant", "content": response})
                        