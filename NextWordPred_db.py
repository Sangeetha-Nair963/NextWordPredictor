import sqlite3
import json
from collections import OrderedDict

class Predictor_Database:
    def __init__(self, table):
        """Initialize db class variables"""
        self.table = table
        self.connection = sqlite3.connect('NextWordPredictor.db', check_same_thread=False)
        print ('Opened database successfully')
        cursor = self.connection.execute('''CREATE TABLE IF NOT EXISTS WORD_RECORD
            (WORD NOT NULL, COUNT INT NOT NULL, FLAG NOT NULL);''')
        cursor.close()
        cursor = self.connection.execute('''CREATE TABLE IF NOT EXISTS LOGIN
            (EMAIL_ID NOT NULL, PASSWORD NOT NULL);''')
        cursor.close()
        cursor = self.connection.execute('''CREATE TABLE IF NOT EXISTS FEEDBACK
            (PREDICTION_ACCURACY NOT NULL, RECOMMEND NOT NULL, HOW_APP NOT NULL, SPEECH_RECOGNITION NOT NULL);''')
        cursor.close()
        print ('Table created successfully')

    def close(self):
        """close sqlite3 connection"""
        self.connection.close()
        print ('Closed database successfully')

    def createTable(self):
        self.connection.execute('''CREATE TABLE IF NOT EXISTS WORD_RECORD
            (WORD NOT NULL, COUNT INT NOT NULL, FLAG INT NOT NULL);''')
        print ('Table created successfully')

    def readWithWhere(self, where):
        sql = "SELECT * FROM {} WHERE {} = '{}'".format(self.table,where['key'],where['value'])
        print(sql)
        cursor = self.connection.execute(sql)
        return cursor.fetchone()
    
    def readFeedback(self):
        sql = 'SELECT * FROM FEEDBACK'
        cursor = self.connection.execute(sql)
        return cursor.fetchone()
    
    def readAll(self):
        sql = 'SELECT WORD, COUNT, FLAG FROM {}'.format(self.table)
        print(sql)
        cursor = self.connection.execute(sql)
        rows = '['
        word = '{'
        cnt = '{'
        flag = '{'
        count = 0
        for row in cursor:
            if count > 0:
                word = word + ','
                cnt = cnt + ','
                flag = flag + ','
            word = word + '"' + str(count) + '":"' + row[0] +'"'
            cnt = cnt + '"' + str(count) + '":' + str(row[1])
            flag = flag + '"' + str(count) + '":"' + row[2] +'"'
            count=count+1
        word = word + '}'
        cnt = cnt + '}'
        flag = flag + '}'
        rows = '{ "WORD":'+ word + ',"COUNT":'+ cnt + ',"FLAG":'+flag+'}'
        OrderedData = json.loads(rows, object_pairs_hook=OrderedDict)
        return json.dumps(OrderedData)
        
    def insert(self, row):
        bindings = '('
        keys = '('
        values = []
        i = 0
        for key, value in row.items():
            bindings += '?'
            keys += key
            values.append(value)
            i += 1
            if i != (len(row)):
                bindings += ', '
                keys += ', '
        bindings += ')'
        keys += ')'
        sql = 'INSERT INTO {} {} VALUES {}'.format(self.table,keys, bindings)
        print(sql, values)
        self.connection.execute(sql, values)
        print("Inserted successfully")
        self.connection.commit()

    def update(self, row, where):
        keys = ''
        values = []
        i = 0
        for key, value in row.items():
            keys += key + ' = ?'
            values.append(value)
            i += 1
            if i != len(row):
                keys += ', '
        sql = "UPDATE {} SET {} WHERE {} = '{}'".format(self.table, keys, where['key'], where['value'])
        print(sql, values)
        self.connection.execute(sql, values)
        self.connection.commit()

    def updInsertDb(self,where,row,flag):
        data = self.readWithWhere(where)
        if(data != None):
            counter = data[1]
            counter = counter + 1
            self.update(dict(COUNT=counter,FLAG=flag), where)
        else:
            self.insert(row)