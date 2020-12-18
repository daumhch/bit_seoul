# pip install pymssql
import pymssql as ms
# import numpy as np

# conn = ms.connect(server='localhost', user='gema2', password='1234', database='mydb')
conn = ms.connect(server='127.0.0.1', user='bit2', password='1234', database='bitdb')
# conn = ms.connect(server='127.0.0.1', user='sa', password='1234', database='test01')

cursor = conn.cursor()

cursor.execute('SELECT * FROM iris2;')
# cursor.execute('SELECT * FROM iris_ys;')


row = cursor.fetchone()
# print(type(row))        # tuple

while row:
    print("첫컬럼=%s, 둘컬럼=%s" %(row[0], row[1]))
    # print(row)
    row = cursor.fetchone()

conn.close()
