import pyodbc as pyo

server = '127.0.0.1'
database = 'bitdb'
username = 'bit2'
password = '1234'

conn = pyo.connect('DRIVER={ODBC Driver 17 for SQL Server}; SERVER=' +server+
                   '; PORT=1433; DATABASE=' +database+
                   '; UID=' +username+
                   '; PWD=' +password )
curser = conn.cursor()

tsql = 'SELECT * FROM iris2;'

with curser.execute(tsql):
    row = curser.fetchone()

    while row :
        print(str(row[0]) + " " + str(row[1]) + " " + str(row[2]) + " " +
              str(row[3]) + " " + str(row[4]))
        row = curser.fetchone()
conn.close()        

