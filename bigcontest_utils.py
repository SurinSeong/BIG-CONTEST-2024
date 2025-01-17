import streamlit as st
import torch

######################embedding###############################
from transformers import AutoTokenizer, AutoModel
from langchain_huggingface import HuggingFaceEmbeddings



def clear_chat_history():
    st.session_state.messages = [{"role": "assistant", "content": "어떤 식당을 찾으시나요?"}]
    
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
