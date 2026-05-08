# AI-Competition

## 중도포기 금지 X 포기하지 말고 끝까지 
- 

## 팀원
- 최민식
- 임현진
- 홍우석
- 박예진
- 권혁상

## 일정
- 2026/05/12 >> 문제 정의
  
- 2026/05/21 >> 베이스라인 모델 생성
  
- 2026/06/05 >> 모델 성능 검증
  
- 2026/06/13 >> 배포
  
- 2026/06/21 >> 발표자료 완료
  
- 2026/06/27 >> 제출
  

## 사용 기술
- Python
- Colab
- Scikit-learn

## 폴더 구조

-data	데이터 창고 >> csv 파일 넣는 

예시 >> 

data/
├── train.csv
├── test.csv
└── sample_submission.csv

-notebooks	코랩/ipynb >> 분석/실험용 ipynb 파일 넣는 곳

예시 >>

notebooks/
├── 01_EDA.ipynb
├── 02_Preprocessing.ipynb
└── 03_Modeling.ipynb

-src	전처리/모델 코드 >> 재사용할 코드 저장하는 곳 / 코랩만 사용하면 코드 관리가 어렵기 때문에 중요한 코드 따로 빼는 거임

예시 >> 

src/
├── preprocess.py
├── train.py
└── predict.py


-models	저장 모델 >> 학습 끝난 AI 저장

예시 >> 

models/
├── model_v1.pkl
└── model_v2.pkl


-app	Streamlit >> 서비스 화면 만드는 곳 / 사용자 화면


-results	결과 이미지/CSV >> 결과 저장 / 그래프, csv 예측 결과, 발표자료용 이미지 등

예시 >> 

results/
├── confusion_matrix.png
├── score.csv
└── feature_importance.png
