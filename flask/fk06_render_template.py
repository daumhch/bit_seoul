from flask import Flask, render_template

app = Flask(__name__)

@app.route('/')
def index():
    return render_template('index.html')        # render_template : 현재 작업하고 있는 파일 하단에 존재해야함

@app.route('/user/<name>')
def user(name):
    return render_template('user.html', name=name)  # user.html에 name을 보냄

if __name__ == "__main__":
    app.run(host='127.0.0.1', port=5000, debug=False)



