from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from konlpy.tag import Twitter

app = Flask(__name__)

######## load classifer
######## 앱 재시작시 마다 원본으로 복구 됨
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects','classifier.pkl'), 'rb'))

######## load vect
twitter = Twitter()
def tw_tokenizer(text):
    tokens_ko = twitter.morphs(text)
    return tokens_ko
tfidf_matrix_train = pickle.load(open(os.path.join(cur_dir,'pkl_objects','tfidf_matrix_train.pkl'), 'rb'))

######## access db
db = os.path.join(cur_dir, 'brunch_network.db')

######## input text classify
######## 반환값 : 글 카테고리, 확률
def classify(document):
    label = {0:'IT_트렌드', 1:'감성_에세이', 2:'건강·운동', 3:'건축·설계',
                  4:'그림·웹툰', 5:'글쓰기_코치', 6:'디자인_스토리', 7:'멋진_캘리그래피', 8:'멘탈_관리_심리_탐구',
                   9:'문화·예술', 10:'뮤직_인사이드', 11:'사랑·이별', 12:'사진·촬영', 13:'쉽게_읽는_역사',
                  14:'스타트업_경험담', 15:'시사·이슈', 16:'오늘은_이런_책', 17:'요리·레시피', 18:'우리집_반려동물',
                  19:'육아_이야기', 20:'인문학·철학', 21:'지구한바퀴_세계여행', 22:'직장인_현실_조언', 23:'취향저격_영화_리뷰'}
    X = tfidf_matrix_train.transform([document]) # input text tfidf 변환 ## transform
    y = clf.predict(X)[0] # return predicted label
    proba = np.max(clf.predict_proba(X)) # return probability of labels
    return label[y], proba

######## train text if wrong answer
######## 예측한 라벨이 틀릴경우 다시 학습시킴
def train(document, y):
    X = tfidf_matrix_train.transform([document])
    clf.partial_fit(X, [y]) # partial_fit => 부분학습?

########
def sqlite_main(path, document, answer, y, correction_label ):
    conn = sqlite3.connect('brunch_network.db')
    c = conn.cursor()
    c.execute("INSERT INTO main_table (text,answer,pred_label, correction_label, date)"\
    " VALUES (?, ?, ?, ?, DATETIME('now'))", (document,answer,y,correction_label))
    conn.commit()
    conn.close()

#############################
######## 플라스크 ############
#############################
#############################
class InputText(Form):
    input_text = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)]) # 글자 최소 15자 이상 입력

@app.route('/') # 시작 페이지
def index():
    form = InputText(request.form) # text 입력 폼 -> input_text.html로 출력
    return render_template('input_text.html', form=form)


@app.route('/results', methods=['POST'])
def results(): # 결과반환 페이지로
    form = InputText(request.form)
    if request.method == 'POST' and form.validate(): # intput text의 조건이 갖추어졌다면
        text = request.form['input_text']
        y, proba = classify(text)
        # sqlite_main(db, text, y) # 입력한 text, 예측결과 db로
        return render_template('results.html', # 결과페이지로 전환
                                content=text,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('input_text.html', form=form) # 조건 갖추어지지 않았을시 입력페이지

@app.route('/thanks', methods=['POST']) # 맞춤 틀림
def feedback():
    form = InputText(request.form)
    feedback = request.form['feedback_button']
    text = request.form['text']
    prediction = request.form['prediction']
    probability = request.form['probability']

    if feedback == 'correct':
        answer = 1
        correction_label = None
        sqlite_main(db, text, answer, prediction, correction_label )
        return render_template('thanks.html')

    elif feedback == 'incorrect':
        return render_template('incorrect_feedback.html',content=text,prediction=prediction,probability=probability)

@app.route('/correction', methods=['POST']) # 틀림 -> 피드백
def correction_category():
    form = InputText(request.form)
    correction = request.form['correction_button']
    text = request.form['text']
    prediction = request.form['prediction']

    class_condition = {'지구한바퀴_세계여행':0, '그림·웹툰':1, '시사·이슈':2,
    'IT_트렌드':3, '사진·촬영':4, '취향저격_영화_리뷰':5, '오늘은_이런_책':6,
    '뮤직_인사이드':7, '글쓰기_코치':8, '직장인_현실_조언':9, '스타트업_경험담':10,
    '육아_이야기':11, '요리·레시피':12, '건강·운동':13,'멘탈_관리_심리_탐구':14,
    '디자인_스토리' :15, '문화·예술':16, '건축·설계':17, '인문학·철학':18,
    '쉽게_읽는_역사':19, '우리집_반려동물':20,'멋진_캘리그래피':21, '사랑·이별':22,
    '감성_에세이':23} ## 사랑이별 + 감성에세이,  디자인_스토리 + 멋진_캘리그래피

    answer = 0
    y = prediction
    correction_label = class_condition[correction]
    sqlite_main(db, text, answer, y, correction_label)
    # train(text, y)

    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True) # debug=True
