from flask import Flask
app = Flask(__name__)

from flask import make_response

@app.route('/')
def index():
    response = make_response('<h1> 잘 따라 치시오!!! </h1>')
    response.set_cookie('answer', '42')             # 'answer' :쿠키의 이름 / '42' : 값
    return response

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=False)

    
'''            cookie          session
------------------------------------------
저장위치 :    클라이언트     /    서버
저장형식 :   텍스트 형식     /  object형
종료시점 : 쿠키 저장 시 설정 / 브라우저 종료시
          (기간 만료시 삭제)/      삭제
                           / (기간 지정 가능)
  자원   : 클라이언드의     / 서버 자원 사용
            자원 사용 
  속도   :     빠름        /      느림        : 상대적
  보안   :     나쁨        /      좋음        : 상대적
'''