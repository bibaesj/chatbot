import os
import streamlit as st
import warnings
from sentence_transformers import SentenceTransformer
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import faiss
import numpy as np

# 경고 메시지 무시
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=DeprecationWarning)

# 페이지 설정
st.set_page_config(page_title="화학 물질 화재 사고 전문 챗봇", page_icon="🤖")
st.title("화학 물질 화재 사고 전문 챗봇")

# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []

# 문서를 로드하고 벡터 저장소에 추가하는 함수
@st.cache_resource
def load_documents(directory):
    documents = []
    filenames = []
    for filename in os.listdir(directory):
        if filename.endswith(".txt"):
            filepath = os.path.join(directory, filename)
            with open(filepath, 'r', encoding='utf-8') as file:
                text = file.read()
                documents.append(text)
                filenames.append(filename)
    return documents, filenames

# 벡터 저장소 로드
@st.cache_resource
def load_vector_store():
    # Hugging Face Sentence Transformer 모델을 사용한 임베딩
    embeddings = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')

    # 인풋 파일 로드
    documents, filenames = load_documents("/Users/macbook/Desktop/streamlit")

    # 문서 임베딩 생성
    document_embeddings = embeddings.encode(documents)  # 텍스트를 임베딩
    
    # FAISS 인덱스 생성
    index = faiss.IndexFlatL2(document_embeddings.shape[1])
    index.add(np.array(document_embeddings).astype(np.float32))  # FAISS에 임베딩 추가
    
    return index, documents, filenames

# 벡터 저장소 로드
index, documents, filenames = load_vector_store()

# GPT 모델과 토크나이저 로드
@st.cache_resource
def load_gpt_model():
    model = GPT2LMHeadModel.from_pretrained("distilgpt2")
    tokenizer = GPT2Tokenizer.from_pretrained("distilgpt2")
    return model, tokenizer

gpt_model, gpt_tokenizer = load_gpt_model()

# 메시지 표시
for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])

# 사용자 입력 처리
if prompt := st.chat_input("질문을 입력하세요"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    # 입력을 임베딩하고 유사한 문서 검색
    query_embedding = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').encode([prompt])
    D, I = index.search(np.array(query_embedding).astype(np.float32), k=3)
    
    # 검색된 문서들을 기반으로 GPT 모델에서 답변 생성
    context = " ".join([documents[i] for i in I[0]])
    
    # 입력 텍스트 길이 제한 (GPT-2의 최대 입력 토큰은 1024 토큰)
    max_context_length = 500  # 텍스트 길이를 500자로 제한
    limited_context = context[:max_context_length]  # 텍스트 길이를 제한
    
    input_text = f"맥락: {limited_context}\n\n질문: {prompt}\n\n답변:"
    input_ids = gpt_tokenizer.encode(input_text, return_tensors="pt")

    # max_new_tokens를 사용하여 모델이 생성할 토큰 수를 제한
    output = gpt_model.generate(input_ids, max_new_tokens=150)
    
    response_text = gpt_tokenizer.decode(output[0], skip_special_tokens=True)

    with st.chat_message("assistant"):
        st.markdown(response_text)
    
    st.session_state.messages.append({"role": "assistant", "content": response_text})
