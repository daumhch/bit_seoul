from flask import Flask

app = Flask(__name__)

@app.route("/<name>")        # /<name> : 아무거나 // 안써주면 Not Found
def user(name):
    return '<h1>Hello, %s !!!</h1>' %name       # Hello, 아무거나

@app.route("/user/<name>")
def user2(name):
    return '<h1>Hello, user/%s !!!</h1>' %name

if __name__ == '__main__' :
    app.run(host='127.0.0.1', port=5001)
    