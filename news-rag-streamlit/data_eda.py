from langchain.chains import LLMChain
from langchain_core.prompts import ChatPromptTemplate
import json
from dotenv import load_dotenv
import os
from langchain_openai import ChatOpenAI
from langchain_core.messages import HumanMessage

load_dotenv()

# 모델 설정
model = ChatOpenAI(model="gpt-4o-mini")

# 폴더 내에 있는 파일 모두 가져오기
path = 'crawlingfile-challenge/'
file_list = os.listdir(path)

# 폴더 내의 데이터 읽어오기
json_data = []
n = len(file_list)

for i in file_list[n-1:n]:
    with open('crawlingfile-challenge/'+i, 'r', encoding='utf-8') as f:
        file = json.load(f)
        json_data.append(file)


# 프롬프트 생성
summary_prompt = ChatPromptTemplate.from_messages([
    ('system', 'Answer the summary of the content.'),
    ('user', '{content}')
])

# LLM 체인 구성
llm_chain = summary_prompt | model


# 요약 내용 정리
for i in range(0, len(json_data)):
    for j in range(0, len(json_data[i])):
        data = json_data[i][j]['content']

        # invoke의 반환 값에서 'text' 키를 추출
        summary = llm_chain.invoke({"content": data}).content
        # 기존 'content'에 요약된 텍스트 저장
        json_data[i][j]['content'] = summary

with open('data.json', 'w', encoding='utf-8') as file:
    json.dump(json_data, file, ensure_ascii=False, indent=4)
