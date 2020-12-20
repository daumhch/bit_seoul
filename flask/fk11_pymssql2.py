import pymssql as ms

conn = ms.connect(server='127.0.0.1', user='bit2', password='1234', database='bitdb')

cursor = conn.cursor()

cursor.execute('SELECT * FROM ADDRESS;')

row = cursor.fetchone()

while row:
    print("첫컬럼=%s, 둘컬럼=%s" %(row[0], row[1]))
    row = cursor.fetchone()

conn.close()
