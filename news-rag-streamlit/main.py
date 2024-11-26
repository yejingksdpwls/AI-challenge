from langchain_core.runnables import RunnablePassthrough
from uuid import uuid4
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.docstore.document import Document
from langchain.document_loaders import TextLoader
from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
import json
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage
import streamlit as st

load_dotenv()

# 모델 설정
model = ChatOpenAI(model="gpt-4o-mini")


# 폴더 내의 데이터 읽어오기
json_data = []


with open('data.json', 'r', encoding='utf-8') as f:
    json_data = json.load(f)



# JSON 데이터를 Document로 변환

def nested_json_to_documents(json_data):
    documents = []

    # 중첩된 리스트를 순회하며 평탄화
    for outer_list in json_data:  # 최상위 리스트 순회
        for entry in outer_list:  # 중첩 리스트 순회
            # 필요한 필드를 결합해 하나의 문서로 구성
            content = (
                f"Title: {entry['title']}\n"
                f"Description: {entry['description']}\n"
                f"Content: {entry['content']}\n"
                f"URL: {entry['url']}\n"
                f"Date: {entry['date']}"
            )
            metadata = {
                'title': entry.get('title', ''),
                'url': entry.get('url', ''),
                'date': entry.get('date', ''),
            }
            # Document 생성
            documents.append(Document(page_content=content, metadata={
                "source": entry["url"], "date": entry["date"]}))

    return documents


documents = nested_json_to_documents(json_data)


# OpenAI Embeddings 사용
embeddings = OpenAIEmbeddings()

# 고유 id 생성
uuids = [str(uuid4()) for _ in range(len(documents))]

# JSON 데이터를 벡터 데이터베이스로 저장
vector_store = FAISS.from_documents(documents=documents, ids=uuids, embedding=embeddings, docstore=InMemoryDocstore())


# 유사성 검색 retriever 정의
retriever = vector_store.as_retriever(search_type='similarity', search_kwargs={'k': 1})


# 프롬프트 템플릿 정의
contextual_prompt = ChatPromptTemplate.from_messages([
    ("system", "Answer the question using only the following context."),
    ("user", "Context: {context}\\n\\nQuestion: {question}")
])


class DebugPassThrough(RunnablePassthrough):
    def invoke(self, *args, **kwargs):
        output = super().invoke(*args, **kwargs)
        print("Debug Output:", output)
        return output
# 문서 리스트를 텍스트로 변환하는 단계 추가


class ContextToText(RunnablePassthrough):
    def invoke(self, inputs, config=None, **kwargs):  # config 인수 추가
        # context의 각 문서를 문자열로 결합
        context_text = "\n".join(
            [doc.page_content for doc in inputs["context"]])
        return {"context": context_text, "question": inputs["question"]}


# RAG 체인에서 각 단계마다 DebugPassThrough 추가
rag_chain_debug = {
    "context": retriever,                    # 컨텍스트를 가져오는 retriever
    "question": DebugPassThrough()        # 사용자 질문이 그대로 전달되는지 확인하는 passthrough
} | DebugPassThrough() | ContextToText() | contextual_prompt | model


# streamlit 실행
# 세션 상태 초기화
if "messages" not in st.session_state:
    st.session_state.messages = []  # 대화 기록을 저장

# 대화 입력
prompt = st.chat_input("Enter your message.")
if prompt:
    # 사용자의 메시지를 기록
    st.session_state.messages.append({"role": "user", "content": prompt})

    # 모든 이전 메시지를 기반으로 응답 생성
    conversation_history = [
        {"role": msg["role"], "content": msg["content"]}
        for msg in st.session_state.messages
    ]
    response = rag_chain_debug.invoke(prompt).content

    # 챗봇의 응답을 기록
    st.session_state.messages.append({"role": "assistant", "content": response})

# 대화 기록 표시
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])        