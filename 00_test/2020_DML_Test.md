
# 2020-1 Distributed Machine Learning
## 수강능력시험

1. 정답은 마크다운으로 작성, 화요일 23시 59분까지 분산머신러닝처리 repository에 pull request 할 수 있도록 합니다.(Answer_Q1_학번(사번).md)
2. 인터넷을 찾아보지 않고, 최대한 자기 실력으로 문제를 풀어 주시기 바랍니다
3. To be Honest!

### Q1. 다음 코드를 작동할 수 있도록 고치시오
 - Hint : Text Mining의 데이터 분석 프로세스


```python
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.pipeline import Pipeline

categories = [
    'alt.atheism',
    'talk.religion.misc',
]
data = fetch_20newsgroups(subset='train', categories=categories)

pipeline = Pipeline([
    ('tfidf', TfidfTransformer()),
    ('vect', CountVectorizer()),
    ('clf', SGDClassifier()),
])

parameters = {
    'vect__max_df': (0.5, 0.75, 1.0),
    'vect__ngram_range': ((1, 1), (1, 2)),  # unigrams or bigrams
    'clf__max_iter': (20,),
    'clf__alpha': (0.00001, 0.000001),
    'clf__penalty': ('l2', 'elasticnet'),
}

grid_search = GridSearchCV(pipeline, parameters, n_jobs=-1, verbose=1)

grid_search.fit(data.data, data.target)
```

### Q2. 신민철은 최근에 SVM을 이용하여 붓꽃 분류 모델을 작성 후 공유하였다. 완성된 머신 러닝 모델을 불러와서 시험해보세요
- smc_iris_model.joblib 파일을 불러와서 iris data에서 predict를 진행해보세요.
- iris 데이터를 다운로드 받는 코드
> from sklearn import datasets<br>
X, y = datasets.load_iris(return_X_y=True)


```python

```

### Q3. 신민철은 영화 박스오피스 데이터를 수집하기 위해, 영화진흥위원회 OPENAPI를 이용하여 데이터를 수령하기로 하였습니다. 제공되는 설명을 읽고 API 에서 2019년 12월 1일부터 12월 31일까지의 박스오피스 데이터를 받아 데이터프레임 형태로 수집하세요

* 주소 : http://www.kobis.or.kr/kobisopenapi/homepg/apiservice/searchServiceInfo.do?serviceId=searchMovieList
(가입이 필요합니다)

* 요청 시 주의할 파라미터: repNationCd = 'K'(한국 영화만 조회하세요)

* 완성된 데이터프레임의 header : movieNm(영화이름), showRange(박스오피스 조회 일자), salesAmt(해당 일의 매출액), audiCnt(일일 관객 수), scrnCnt(스크린 수)


```python

```

### Q4. 파이썬에서 모듈과 클래스의 차이는 무엇인가요? 한 마디로 설명해 주세요.


```python

```

### Q5. AWS EC2, Google Cloud Compute Engine, MS Azure VMs 중 한 곳에 자신의 컴퓨팅 노드를 만들고, 쉘에 접속한 사진을 찍어서 같이 업로드해주세요.
**(과금주의!!!)반드시 사진을 찍은 후에는 만든 VM을 삭제하셔야 과금되지 않습니다.**


```python

```
