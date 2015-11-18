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

            self.conn.commit()
        except sqlite3.Error as e:
            print(e)

        try:
            self.cur.execute("""create table validation_scores
             (datetime INTEGER, accuracy REAL, parameters TEXT, 
              train_length INTEGER, notes TEXT)""")

            self.conn.commit()
        except sqlite3.Error as e:
            print(e)

    def open_conn(self):
        if not self.conn:
            self.conn = sqlite3.connect(self.dbfilepath)
            self.cur = self.conn.cursor()

    def close_conn(self):
        if self.conn:
            self.conn.close()
        self.conn = None
        self.cur = None

    def check_conn(self):
        if not self.conn or not self.cur:
            self.open_conn()
            
    def save_validation_score(self, clf, accuracy, train_length, notes=""): 
        self.check_conn()
       now = int(datetime.datetime.now().timestamp())
        self.cur.execute("""INSERT INTO validation_scores VALUES
             (?,?,?,?,?)""", (now, accuracy, str(clf), train_length, notes))
        self.conn.commit()

    def save_test_predictions(self, predictions, clf, train_length, notes=""):
        self.check_conn()
        now = int(datetime.datetime.now().timestamp())
        fname = str(now) + ".csv"
        self.cur.execute("""INSERT INTO submissions VALUES
             (?,?,?,?,?)""", (now, fname, str(clf), train_length, notes))
        self.conn.commit()
        predictions.to_csv(self.output_folder + "/" + fname, index=False)

    def get_max_validation_score(self):
        self.check_conn()
        for row in c.execute("SELECT * FROM validation_scores group by accuracy having max(accuracy)"):
            print(row)
