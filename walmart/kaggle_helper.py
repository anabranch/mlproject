import sqlite3
import datetime


class KaggleHelper:
    def __init__(self, dbfilepath, output_folder):
        self.dbfilepath = dbfilepath
        self.conn = None
        self.cur = None
        self.output_folder = output_folder
        self.open_conn()
        try:
            self.cur.execute("""create table submissions
             (datetime INTEGER, filename TEXT, parameters TEXT, 
              train_length INTEGER, notes TEXT)""")
        except sqlite3.Error as e:
            print(e)
        self.conn.commit()

    def open_conn(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.dbfilepath)
            self.cur = self.conn.cursor()

    def close_conn(self):
        if self.conn:
            self.conn.close()
        self.conn = None
        self.cur = None

    def save_predictions(self, predictions, clf, train_length, notes):
        if not self.conn or not self.cur:
            self.open_conn()
        now = int(datetime.datetime.now().timestamp())
        fname = str(now) + ".csv"
        self.cur.execute("""INSERT INTO submissions VALUES
             (?,?,?,?,?)""", (now, fname, str(clf), train_length, notes))
        self.conn.commit()
        predictions.to_csv(self.output_folder + "/" + fname, index=False)
        self.close_conn()
