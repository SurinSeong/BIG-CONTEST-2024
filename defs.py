import streamlit as st 
import json
##################################################################api
import google.generativeai as genai
##################################################################vectordb
from langchain_chroma import Chroma
##################################################################embedding
from langchain_huggingface import HuggingFaceEmbeddings


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

###############################[채팅내용 reset]##################################
def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당을 찾으시나요?"}]
    
###############################[llm 함수]##################################
def get_llm():
    # Create the mod
    llm = genai.GenerativeModel(model_name="gemini-1.5-flash",
                            generation_config={"temperature": 0,
                                                "max_output_tokens": 5000,},)
    return llm   

###############################[체크표시 필터링]##################################
def search_restaurants(query, local_jeju_city=False, local_seogwipo_city=False):
    filtered_results = []
    
    # 'metadata' 변수는 함수 외부에서 가져옵니다
    global metadata
    
    if local_jeju_city:
        for meta in metadata['metadatas']:  # 'metadatas' 리스트 순회
            if isinstance(meta, dict) and '제주시' in meta.get('address', ''):  # 'address'가 포함된 데이터만 처리
                filtered_results.append(meta)
    elif local_seogwipo_city:
        for meta in metadata['metadatas']:
            if isinstance(meta, dict) and '서귀포시' in meta.get('address', ''):  # 'address'가 포함된 데이터만 처리
                filtered_results.append(meta)
    else:
        for meta in metadata['metadatas']:
            if isinstance(meta, dict):  # 딕셔너리인 경우만 처리
                filtered_results.append(meta)

    return filtered_results

#################################### 카테고리 분류 #####################################################
def category_classification(query):
    
    classification_system_prompt = \
    """
    당신은 사용자의 입력을 '카테고리'에 맞게 분류하는 역할을 맡았습니다.
    사용자의 입력은 제주도 맛집에 대한 질문입니다. 사용자의 입력을 분석해 다음 '카테고리' 중 하나로 분류합니다:
    ---
    [카테고리]
    검색형
    추천형
    ---
    만약, 위의 '카테고리' 중 어느 곳에도 포함되지 않는다면, '기타' 로 분류해주세요.
    만약, 제주도가 아닌 다른 지역을 언급한다면 '기타' 로 분류해주세요.
    만약, 식당, 음식점 관련된 내용을 언급하지 않는다면 '기타' 로 분류해주세요.
    지명, 현지인 비중, 위치, 가게명, 동행자, 방문 목적과 관련되지 않은 음식점에 대한 추천 질문은 '기타' 로 분류해주세요.
    다른 설명 없이 '카테고리'만으로 응답해주세요.
    사용자가 하는 질문을 철저하게 분석해, '검색형', '추천형' 또는 '기타' 로 분류하세요.
    """

    # 모델 생성
    model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                  generation_config={"temperature": 0,
                                                     "max_output_tokens": 5000,},
                                  system_instruction=classification_system_prompt)

    # LLM에게 카테고리를 분류하도록 프롬프트 작성
    classification_template = \
    """
    - 검색형 질문의 경우: 사용자가 특정 데이터를 요청하는 질문입니다. '이용건수', '비중', '구간'과 같은 데이터를 요구하는 경우 검색형 질문으로 분류하세요. '비중이 높은', '이용 건수가 적은'과 같이 특정 수치나 비율을 요청하는 경우도 검색형 질문으로 분류하세요.
    예: '제주시 노형동에 있는 단품요리 전문점 중 이용건수가 상위 10%에 속하고 현지인 이용 비중이 가장 높은 곳은?'
    - 추천형 질문의 경우: 사용자가 특정 상황에 맞는 장소를 추천해 달라는 질문입니다. '가족과 함께' 또는 '비지니스 자리'와 같은 단어가 포함된 경우 추천형 질문으로 분류하세요.
    예: '가족과 함께 갈 만한 횟집 추천해줘.'

    검색형 질문일 경우에는 반드시 특정한 데이터 지표('이용건수', '비중', '구간')와 연관된 질문이어야 하며, 그렇지 않으면 추천형으로 분류하세요.

    --
    [카테고리 가이드 라인]

    추천형 카테고리 분류 예시:
    '12시에서 13시 사이에 영업 중인 카페를 알려주세요.', '새벽3시에 운영하는 음식점은?'
    이 질문은 특정 시간대에 이용 가능한 카페를 알려달라는 내용으로, 사용자가 선택할 수 있는 대안들 중에서 추천을 요청하는 질문입니다. 따라서 이 질문은 추천형 카테고리로 분류됩니다.

    검색형 카테고리 분류 예시:
    '12시에서 13시 사이에 이용 건수가 가장 적은 카페를 알려주세요.'
    이 질문은 특정 조건(이용 건수가 적은 곳)을 충족하는 카페를 찾아달라는 구체적인 정보를 요구하는 내용입니다. 따라서 이 질문은 검색형 카테고리로 분류됩니다.

    기타 카테고리 분류 예시:
    제주도 내 지역이 아닌 경우, 또는 식당, 음식점 관련된 내용이 아닌 경우 기타 카테고리로 분류합니다.
    --
    사용자 질문: {query}

    JSON 형식으로 반환하세요:
    {{
          'Classification': '검색형',
          'query': '{query}'
    }}

    {{
        'Classification': '추천형',
        'query': '{query}'
    }}

    {{
        'Classification': '기타',
        'query': '{query}'
    }}
    """

    prompt = classification_template.format(query=query)
    response = model.generate_content(prompt, generation_config={'response_mime_type':'application/json'})
    classification = response.text.strip()
    classification_json = json.loads(classification)
    return classification_json

###############################################검색형 함수###############################################
# LLM을 이용한 키워드 추출 함수
def extract_keywords_from_text(query):
    # 미리 genai에게 페르소나를 준다.
    keywords_system_prompt = \
    """
    당신은 데이터 분석 전문가로서, 사용자가 입력한 문장에서 중요한 정보를 추출하는 역할을 맡고 있습니다.
    주어진 키 목록을 기준으로 문장을 분석하고, 필요한 정보를 JSON 형식으로 반환해 주세요.

    """
    # 모델 생성
    model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                  generation_config={"temperature": 0,
                                                     "max_output_tokens": 5000,},
                                  system_instruction=keywords_system_prompt)


    # 제미나이 LLM 프롬프트 생성
    keywords_template = \
    """
    문장에서 중요한 정보를 JSON 형식으로 추출해주세요. 키는 아래의 리스트 중에서 문맥상 가장 적합한 항목을 사용해주세요.

    keys = [
        '가맹점명', '개설일자', '업종', '주소', '이용건수구간', '이용금액구간', '건당평균이용금액구간',
        '월_이용건수비중', '화_이용건수비중', '수_이용건수비중', '목_이용건수비중', '금_이용건수비중', '토_이용건수비중',
        '일_이용건수비중', '05-11시_이용건수비중', '12-13시_이용건수비중', '14-17시_이용건수비중',
        '18-22시_이용건수비중', '23-04시_이용건수비중', '현지인_이용건수비중', '남성_이용건수비중',
        '여성_이용건수비중', '10-20대_회원비중', '30대_회원비중', '40대_회원비중', '50대_회원비중',
        '60대이상_회원비중'
    ]

    규칙:
    1. 비중으로 끝나는 항목의 값이 "높은" 것을 언급하면 상위 1개를 나타내는 '상위 1개' 형식으로 표시하고, "낮은" 것을 언급하면 하위 1개를 나타내는 '하위 1개' 형식으로 표시해주세요.
    2. 구간으로 끝나는 항목(예: '이용건수구간', '이용금액구간', '건당평균이용금액구간')은 다음과 같은 값으로 변환해주세요:
        - '상위 10%' : '상위 10% 이하',
        - '10~25%' : '10~25%',
        - '25~50%' : '25~50%',
        - '75~90%' : '75~90%',
        - '90% 미만' : '90% 초과'
    3. 주소가 언급되면 key는 '주소', value는 해당 주소로 지정해주세요.
    4. 업종이 입력되면,
        ['가정식', '커피', '분식', '단품요리 전문', '치킨', '중식', '맥주/요리주점', '양식', '베이커리',
        '아이스크림/빙수', '일식', '샌드위치/토스트', '구내식당/푸드코트', '피자', '떡/한과', '민속주점',
        '햄버거', '동남아/인도음식', '꼬치구이', '패밀리 레스토랑', '차', '도시락', '야식', '부페',
        '도너츠', '스테이크', '기타세계요리', '기사식당', '주스', '포장마차']
    이 중 하나로 변환해주세요. 없다면 유사한 것으로 변환해주세요.
    (예: 업종이 '단품요리 전문점'으로 입력되면 '단품요리 전문'으로 변환해주세요.)
    (예: 업종이 '한식점'으로 입력되면 '가정식', '기사식당'등으로 더 유사한 것으로 변환해주세요.)

    예시:
    사용자 입력: '제주시 노형동에 있는 단품요리 전문점 중 이용건수가 상위 10%에 속하고 현지인 이용 비중이 가장 높은 곳은?'
    출력:
    {{
        "주소": "제주시 노형동",
        "업종": "단품요리 전문점",
        "이용건수구간": 1,
        "현지인_이용건수비중": "상위 1개"
    }}


    사용자 입력: '{query}'
    """

    prompt = keywords_template.format(query= query)
    response = model.generate_content(prompt, generation_config={'response_mime_type':'application/json'})
    keywords = response.text.strip()

    # 키워드를 쉼표로 구분한 결과를 리스트로 변환
    keywords_json = json.loads(keywords)
    return keywords_json

# 조건으로 정답 찾기
def sorted_df(query, df):
  
  json_data = extract_keywords_from_text(query)

  # 주소로 필터링
  if '주소' in json_data:
      filtered_df = df[df['주소'].str.contains(json_data['주소'])]
  else:
      filtered_df = df  # 주소가 없으면 모든 데이터 사용

  # 업종으로 필터링
  if '업종' in json_data:
      filtered_df = filtered_df[filtered_df['업종'] == json_data['업종']]


  # 구간으로 끝나는 모든 키 필터링
  for key, value in json_data.items():
      if key.endswith('구간') and key in filtered_df.columns:
          filtered_df = filtered_df[filtered_df[key] == value]


  # 비중으로 끝나는 모든 키 필터링 (상위 3개 또는 하위 3개)
  for key, value in json_data.items():
      if key.endswith('비중') and key in filtered_df.columns:
          if value == '상위 1개':
              filtered_df = filtered_df.sort_values(by=key, ascending=False)[:1]
          elif value == '하위 1개':
              filtered_df = filtered_df.sort_values(by=key, ascending=False)[-1:]


  store = {
    'question': query,  # 사용자의 질문
    'answer': filtered_df.가맹점명.iloc[0],  # 가맹점명을 정답으로 넣음
    'summary' : filtered_df.요약.iloc[0]
  }
  return store

def search_chain(store, llm):
    # store에서 질문과 답변을 합쳐 하나의 문자열로 변환
    query_string = store['question'] + ' ' + store['answer']

    # 프롬프트 생성
    chat_template = f"""
    You are an expert assistant. Based on the following question and answer, generate a suitable response using the document provided.
    괄호 안에 들어있는 숫자는 비중인데, 그 내용과 질문 내을 포함해서 작성해주세요.
    ex) 질문 : 제주시 한림읍에 있는 카페 중 30대 이용 비중이 가장 높은 곳은?
        응답 : 제주시 한림읍에 있는 카페 중 30대 이용비중이 0.35로 가장 높은 비중을 가진 가게는 **입니다.


    Question: {store['question']}
    Answer: {store['answer']}
    Document: {store['summary']}

    Use the information from the document to create an informative and coherent response.
    사용자 입력: '{query_string}'

    """
    prompt = chat_template.format()
    response = llm.generate_content(prompt)
    generated_text = response.text.strip().replace("**", "").replace("\n", "").replace("  ", " ")
    return generated_text

# search_main
def search_main(query, df):
  store = sorted_df(query, df)
  llm = get_llm()
  search_response = search_chain(store, llm)
  return search_response

############################################ 추천형 검색#########################################
def extract_recommendation_keywords_from_text(question):
    recommendation_keywords_system_prompt = \
    """
    당신은 데이터 분석 전문가로서, 사용자가 입력한 문장에서 중요한 정보를 추출하는 역할을 맡고 있습니다.
    주어진 키 목록을 기준으로 문장을 분석하고, 필요한 정보를 JSON 형식으로 반환해 주세요.
    """

    # 모델 생성
    model = genai.GenerativeModel(model_name="gemini-1.5-flash",
                                  generation_config={"temperature": 0,
                                                     "max_output_tokens": 5000, },
                                  system_instruction=recommendation_keywords_system_prompt)

    # 제미나이 LLM 프롬프트 생성
    recommendation_keywords_template = \
        """
        문장에서 중요한 정보와 키워드를 dict 형식으로 추출해주세요.

        주어진 질문을 `특정가게정보`, `음식점추천` 중 하나로 분류하세요.
        - '특정가게정보'의 경우: 사용자가 특정 음식점을 제시하고, 그 가게에 대한 정보를 물어보는 경우
        예) '비스트로낭이라는 가게에 대한 정보를 알려줘', '비스트로낭이 현지인 맛집이야?'
        - '음식점추천'의 경우: 사용자가 음식을 제시하여 음식점 추천을 원하거나, 또는 누구와, 특정 목적으로, 특정 관광지 주변에 음식점을 찾는 경우
        예) '연인과 함께가기 좋은 횟집 추천해줘.', '성산일출봉 근처(주변)에 어떤 음식점이 있어?'

        해당 질문의 키워드도 함께 추출하세요.
        ---
        사용자 질문: {query}

        JSON 형식으로 반환하세요:

        FORMAT
        {{
            'classification': '특정가게정보' OR "음식점추천,
            'question": {query},
            'name": 가게명,
            'industry": '이탈리아음식',
            'attraction": 관광지명,
            'local": 1 OR 0
            'keyword": ['가게이름', '정보']
        }}

        규칙
        1. '현지인이 자주가는' 또는 '현지인 맛집' 등 '현지인' 키워드가 질문에 포함될 때,
           문맥 상 사용자가 현지인이 많이 가는 음식점을 찾는다면 'local': 1로 설정합니다.
           관광객이 많이가는 음식점을 찾는다면 'local': 0로 설정합니다.
           단, 현지인과 관광객의 키워드가 query에 존재하지 않는다면, 해당 key와 value는 제거한 형식을 반환합니다.

        2. query에 가게명이 제시되어 있지 않는 '추천형' classification의 경우, 해당 key와 value는 제거한 형식을 반환합니다.

        3. query에 관광지명이 제시되어 있지 않는 경우, 'attraction'의 해당 key와 value는 제거한 형식을 반환합니다.

        4. query에 'industry(업종)'이 제시되어 있지 않는 경우, 해당 key와 value는 제거한 형식을 반환합니다.
           단, 'industry'의 value는 반드시 아래의 리스트 항목 내에서 정해져야 합니다.
           industry_list = ['이탈리아음식', '해물,생선요리', '분식', '돈가스', '곱창,막창,양', '치킨,닭강정', '생선회',
                            '아이스크림', '한식', '돼지고기구이', '육류,고기요리', '카페,디저트', '패밀리레스토랑', '백숙,삼계탕',
                            '중식당', '찌개,전골', '와인', '김밥', '피자', '24시뼈다귀탕', '죽', '햄버거', '베이커리',
                            '테이크아웃커피', '해장국', '요리주점', '국밥', '소고기구이', '종합분식', '양식', '카페', '냉면',
                            '가공식품', '순대,순댓국', '이자카야', '베트남음식', '양꼬치', '마라탕', '포장마차', '프랑스음식',
                            '시장', '매운탕,해물탕', '아귀찜,해물찜', '향토음식', '아시아음식', '우동,소바', '한정식', '일식당',
                            '차', '백반,가정식', '국수', '맥주,호프', '과일', '불닭', '브런치', '쌈밥', '장어,먹장어요리',
                            '바(BAR)', '일품순두부', '만두', '칼국수,만두', '닭요리', '라면', '정육식당', '족발,보쌈',
                            '감자탕', '곰탕,설렁탕', '두부요리', '오리요리', '차,커피', '멕시코,남미음식', '비빔밥', '낙지요리',
                            '태국음식', '조개요리', '술집', '주류', '딤섬,중식만두', '전통,민속주점', '닭갈비', '야식',
                            '샤브샤브', '초밥,롤', '양갈비', '밀키트', '샌드위치', '복어요리', '전복요리', '추어탕',
                            '스페인음식', '오뎅,꼬치', '덮밥', '미향해장국', '게요리', '닭볶음탕', '인도음식', '도넛',
                            '전,빈대떡', '식료품', '도시락,컵밥', '생선구이', '케이크전문', '브런치카페', '일본식라면', '막국수',
                            '과자,사탕,초코렛', '떡볶이', '떡카페', '주꾸미요리', '다방', '스파게티,파스타전문', '갈비탕',
                            '퓨전음식', '한식뷔페', '굴요리', '기사식당', '프랜차이즈본사', '달떡볶이', '33떡볶이', '빙수',
                            '반찬가게', '대게요리', '닭발', '떡류제조', '사철,영양탕', '찐빵', '이북음식', '닭',
                            '일식튀김,꼬치', '육류', '뷔페', '그리스음식', '카레', '가야밀면', '돼지고기', '스테이크,립',
                            '핫도그', '음식점', '일공공키친', '떡,한과']
          이 중 하나로 변환해주세요. 없다면 유사한 것으로 변환해주세요.
          (예: '회집 추천해줘.'의 질문을 사용자가 입력한 경우, 'industry'를 '생선회'로 변환해주세요.)

        5. 제주특별시 내 주소가 query에 들어온 경우, keyword에 추가해주세요.
           반드시 지명이 들어오면, keyword에 추가해주세요.
        ---
        예시:
        '비스트로낭이 현지인 맛집이야?'와 같이 특정 가게정보 질문이면,
        Classification에 "특정가게정보"가 입력되고, Keyword에 "비스트로낭", "현지인", "맛집" 키워드가 입력되도록 해주세요.
        추가로 형식에 맞게 추가해주세요. 해당
        {{
            "Classification": "특정가게정보",
            "Question": "비스트로낭이 현지인 맛집이야?",
            "name": 가게명,
            "local": 1,
            "Keyword": ["비스트로낭", "현지인", "맛집"]
        }}
        '가족과 함께가기 좋은 해장국 집 추천해줘'과 같이 특정 정보에 해당하는 음식점 추천을 위한 질문이면,
        Classification에 "음식점추천"가 입력되고, Keyword에 "가족", "해장국", "추천" 키워드가 입력되도록 해주세요.

        {{
            "Classification": "음식점추천",
            "Question": "가족과 함께가기 좋은 해장국 집 추천해줘",
            "industry": "미향해장국" or "국밥',
            "Keyword": [ "가족", "해장국", "추천"]
            }}

        '연인과 함께 가기 좋은 제주시 애월읍 근처 음식점 추천해줘'과 같이 특정 정보에 해당하는 음식점 추천을 위한 질문이면,
        Classification에 "음식점추천"가 입력되고, Keyword에 "연인", "제주시 애월읍", "추천" 키워드가 입력되도록 해주세요.

        {{
            "Classification": "음식점추천",
            "Question": "연인과 함께 가기 좋은 제주시 애월읍 음식점 추천해줘",
            "Keyword": ["연인", "제주시 애월읍", "추천"]
            }}
        ---
        [가이드라인]
        'attraction'과 주소를 정확히 구분해야 합니다.
        '혼자 가기 좋은 조천읍 주변에 음식점 추천해줘'와 같은 query의 경우, '조천읍'은 주소를 의미하고 Keyword에 추가되어야 합니다.
        제주특별시 내 지역이 query에 포함될 경우, 그 또한 Keyword에 추가되어야 합니다.
        제주도 내 주소 또는 지명을 attraction에 포함해서는 안됩니다.
        """

    # 프롬프트 생성
    prompt = recommendation_keywords_template.format(query=question)

    # LLM에 프롬프트 전송
    response = model.generate_content(prompt, generation_config={'response_mime_type':'application/json'})

    # 응답 정리 및 오류 처리
    keywords_text = response.text.strip()

    # JSON 파싱
    recommendation_keywords_json_data = json.loads(keywords_text)
    return recommendation_keywords_json_data

def filter_chroma_db(query, chroma_store):
    
    filtered_result = search_restaurants(query)
    # 질의를 통해 JSON 데이터 추출
    extracted_data = extract_recommendation_keywords_from_text(query)
    
    # 필터링 조건 설정
    filters = {key: extracted_data[key] for key in ["name", "industry", "attraction", "local"] if key in extracted_data}

    if filters:
        # 필터링 실행
        recommendation_filtered_items = [item for item in filtered_result if all(item.get(key) == value for key, value in filters.items())]
    else:
        # 필터링 조건이 없을 경우 유사한 5개의 항목을 검색
        search_results = chroma_store.similarity_search(query, k=5)
        # 검색 결과에서 메타데이터 추출
        recommendation_filtered_items = [result.metadata for result in search_results]

    return recommendation_filtered_items[:5]

# LLM을 이용한 최종 응답 생성
def recommendation_chain(query, llm, recommendation_store):
    # 데이터베이스에서 관련 정보 검색
    db_results = filter_chroma_db(query, recommendation_store)

    # 검색된 데이터를 LLM 프롬프트에 추가
    db_info = json.dumps(db_results, ensure_ascii=False, indent=2)

    # 채팅 히스토리를 기반으로 프롬프트 생성
    recommendation_chat_template = f"""
    사용자의 질문에 대답해 주세요.
    사용자 입력: '{query}'

    다음 정보는 데이터베이스에서 검색된 결과입니다.:
    {db_info}

    이 정보를 바탕으로 사용자에게 응답을 만들어 주세요.
    """

    # LLM에 프롬프트를 전송하고 결과 받기
    response = llm.generate_content(recommendation_chat_template)
    generated_text = response.text.strip().replace("**", "").replace("\n", "").replace("  ", " ")

    return generated_text

# 메인 함수
def recommendation_main(query, chroma_store):
    llm = get_llm()  # LLM 초기화
    recommendation_response= recommendation_chain(query, llm, chroma_store)  # LLM을 사용하여 응답 생성
    return recommendation_response
###############################################기타형 함수###############################################
def other_chain(query, llm):
    if not metadata:
        # 프롬프트 생성
        other_prompt = \
        f"""
        사용자 질문: {query}
        '사용자 질문'은 '다른 질문을 해달라고 유도하는 답변'을 해야 하는 질문입니다.
        ---
        1. 제주도가 아닌 다른 지역의 음식점이나 맛집을 추천해달라는 질문이면, 제주도의 지역 내 음식점, 맛집을 알려줄 수 있다고 답변해주세요.
        예시) 사용자 질문: '강남의 유명한 곱창집 추천해줘.'
            답변: '제주도 외의 지역에 대한 추천을 해드릴 수 없습니다.. 제주도 지역에 대한 음식점, 맛집에 대해서 알려드릴 수 있어요!'

        2. 지명, 현지인 비중, 위치, 가게명, 동행자, 방문 목적과 관련된 질문이 아닌 다른 질문이면, 관련된 질문을 해달라고 답변해주세요.
        예시) 사용자 질문: '스트레스 풀고 싶을 때 먹기 좋은 음식점 추천해줘.'
            답변: '해당 질문에 대한 답변은 어렵습니다. 현지인 맛집, 특정 지역의 맛집, 특정 동행자와 많이 먹는 음식에 대한 질문을 해주세요!'
        ---
        3. 사용자 질문이 앞뒤가 맞지 않거나 모호한 경우, 아래의 예시처럼 답변해주세요.
        예시) 사용자 질문: '피부 탄력에 좋은 음식점 알려줘'
            답변: '죄송합니다, 입력하신 질문은 이해하기 어려워 추천해 드리기 어렵습니다.
            동행자나 여행 목적에 맞는 제주도 음식점을 추천해 드릴 수 있습니다.
            예를 들어, '가족과 함께 가기 좋은 음식점 추천'이나 '해산물이 맛있는 제주도 음식점'과 같은 질문을 입력해 주시면 도움이 될 것 같습니다'

        예시)를 참고해서 '답변'을 생성해주세요.
        """
        # LLM에 프롬프트를 전송하고 결과 받기
        prompt = other_prompt.format()
        response = llm.generate_content(prompt)
        generated_text = response.text.strip()
        return generated_text


def other_main(query):
    llm = get_llm()
    other_response = other_chain(query, llm)
    return other_response
##############################################메인 함수##############################################
def main(query, df):
    classification = category_classification(query)
    if classification['Classification'] == '검색형':
        print('분류 >> 검색형')
        return search_main(query, df)

    elif classification['Classification'] == '추천형':
        print('분류 >> 추천형')
        return recommendation_main(query, recommendation_store)

    else:
        print('분류 >> 기타')
        return other_main(query)












