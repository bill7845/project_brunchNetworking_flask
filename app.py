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
db = os.path.join(cur_dir, 'brunch_text.sqlite')

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
def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO text_db (text, category, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
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
        return render_template('results.html', # 결과페이지로 전환
                                content=text,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('input_text.html', form=form) # 조건 갖추어지지 않았을시 입력페이지

@app.route('/thanks', methods=['POST']) # 피드백
def feedback():
    feedback = request.form['feedback_button']
    text = request.form['text']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(text, y)
    sqlite_entry(db, text, y)
    return render_template('thanks.html')

@app.route('/correction', methods=['POST']) # 피드백
def correction_category():
    correction = request.form['correction_button']
    text = request.form['text']
    prediction = request.form['prediction']

    inv_label = {'negative': 0, 'positive': 1}
    y = inv_label[prediction]
    if feedback == 'Incorrect':
        y = int(not(y))
    train(text, y)
    sqlite_entry(db, text, y)
    return render_template('thanks.html')

if __name__ == '__main__':
    app.run(debug=True) # debug=True
