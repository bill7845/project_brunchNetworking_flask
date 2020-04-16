from flask import Flask, render_template, request
from wtforms import Form, TextAreaField, validators
import pickle
import sqlite3
import os
import numpy as np
from soynlp.tokenizer import LTokenizer
import ltk

# 로컬 디렉토리에서 HashingVectorizer를 임포트합니다
# from vectorizer import vect

app = Flask(__name__)

######## 분류기 준비
cur_dir = os.path.dirname(__file__)
clf = pickle.load(open(os.path.join(cur_dir,'pkl_objects','classifier.pkl'), 'rb'))

# twitter = Twitter()
# def tw_tokenizer(text):
#     tokens_ko = twitter.morphs(text)
#     return tokens_ko

# ltk = LTokenizer()
# def ltk_tokenizer(text):
#   tokens_ltk = ltk.tokenize(text,flatten=True)
#   return tokens_ltk

ltk_tokenizer = ltk.ltk_tokenizer
tfidf_matrix_train = pickle.load(open(os.path.join(cur_dir,'pkl_objects','tfidf_matrix_train.pkl'), 'rb'))

db = os.path.join(cur_dir, 'brunch_text.sqlite')

def classify(document):
    label = {0:'IT_트렌드', 1:'감성_에세이', 2:'건강·운동', 3:'건축·설계',
                  4:'그림·웹툰', 5:'글쓰기_코치', 6:'디자인_스토리', 7:'멋진_캘리그래피', 8:'멘탈_관리_심리_탐구',
                   9:'문화·예술', 10:'뮤직_인사이드', 11:'사랑·이별', 12:'사진·촬영', 13:'쉽게_읽는_역사',
                  14:'스타트업_경험담', 15:'시사·이슈', 16:'오늘은_이런_책', 17:'요리·레시피', 18:'우리집_반려동물',
                  19:'육아_이야기', 20:'인문학·철학', 21:'지구한바퀴_세계여행', 22:'직장인_현실_조언', 23:'취향저격_영화_리뷰'}
    X = tfidf_matrix_train.transform([document])
    y = clf.predict(X)[0]
    proba = np.max(clf.predict_proba(X))
    return label[y], proba

def train(document, y):
    X = tfidf_matrix_train.transform([document])
    clf.partial_fit(X, [y])

def sqlite_entry(path, document, y):
    conn = sqlite3.connect(path)
    c = conn.cursor()
    c.execute("INSERT INTO text_db (text, category, date)"\
    " VALUES (?, ?, DATETIME('now'))", (document, y))
    conn.commit()
    conn.close()

######## 플라스크
class InputText(Form):
    input_text = TextAreaField('',
                                [validators.DataRequired(),
                                validators.length(min=15)])

@app.route('/')
def index():
    form = InputText(request.form)
    return render_template('input_text.html', form=form)

@app.route('/results', methods=['POST'])
def results():
    form = InputText(request.form)
    if request.method == 'POST' and form.validate():
        text = request.form['input_text']
        y, proba = classify(text)
        return render_template('results.html',
                                content=text,
                                prediction=y,
                                probability=round(proba*100, 2))
    return render_template('input_text.html', form=form)

@app.route('/thanks', methods=['POST'])
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

if __name__ == '__main__':
    app.run() # debug=True
