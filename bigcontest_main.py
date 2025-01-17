import streamlit as st

from langchain_chroma import Chroma

from bigcontest_utils import clear_chat_history, load_embedding_model, embed_text

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
    
    # # PNG 이미지 삽입
    # image = Image.open(r'D:\2024_bigcontest\data\이미지\제주도 지도.png')  # 이미지 파일 경로
    # st.image(image, caption='제주도 지도', use_column_width=True)  # 사이드바에 이미지 삽입

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

st.sidebar.button('Clear Chat History', on_click=clear_chat_history)

tokenizer, model, embedding_function = load_embedding_model()

## 저장된 db 불러오기
# ChromaDB 불러오기
search_store = \
Chroma(collection_name='jeju_store_mct_keyword_v4',
       embedding_function=embedding_function,
       persist_directory=r'C:\Users\tjdtn\inflearn-llm-application\big-contest\mct_keyword_v3')





