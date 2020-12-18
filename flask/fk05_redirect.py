from flask import Flask
from flask import redirect

app = Flask(__name__)

@app.route('/')
def index():
    return redirect('http://www.naver.com')     # 해당 홈페이지로 연결해줌.

if __name__ =='__main__' :
    app.run(host='127.0.0.1', port=5000, debug=False)

