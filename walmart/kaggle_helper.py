import sqlite3
import datetime


class KaggleHelper:
    def __init__(self, dbfilepath):
        """Create a SQLITE File, and open the connection"""
        self.dbfilepath = dbfilepath
        self.conn = None
        self.cur = None
        self.open_conn()

        self._generate_metrics_table()

        self.current_run = None

    def open_conn(self):
        """Open a connection if one isn't already open"""
        if not self.conn:
            self.conn = sqlite3.connect(self.dbfilepath)
            self.cur = self.conn.cursor()

    def close_conn(self):
        """Close a connection if one is open"""
        if self.conn:
            self.conn.close()
        self.conn = None
        self.cur = None

    def check_conn(self):
        "Check the connection and open if it's closed"
        if not self.conn or not self.cur:
            self.open_conn()

    def _generate_metrics_table(self):
        """Generate our metrics table"""
        try:
            self.cur.execute("""
            CREATE TABLE metrics
            (pipeline_number INTEGER NOT NULL,
            val_or_test TEXT NOT NULL check(val_or_test="validation" or val_or_test="test"), 
            start_or_end TEXT NOT NULL check(start_or_end="start" or start_or_end="end"),
            time INTEGER NOT NULL,
            classifier TEXT NOT NULL,
            metric TEXT NOT NULL,
            value TEXT NOT NULL, 
            notes TEXT)
            """)

            self.conn.commit()
        except sqlite3.Error as e:
            print(e)

    def start_pipeline(self):
        """Start a pipeline, this should be called at the beginning on a file
        to signify that you're you're going to create a series of metrics
        based on that all lead up to a test_set being produced"""
        self.check_conn()
        now = int(datetime.datetime.now().timestamp())
        self.current_run = now

    def end_pipeline(self):
        """End a pipeline that has been creating in order for us to create a key
        associated with a series of metrics"""
        self.check_conn()
        self.current_run = None

    def _check_pipeline(self):
        """Check if a pipeliny is currently running and create one if not"""
        if self.current_run == None:
            self.start_pipeline()

    def record_metric(self, val_or_test, start_or_end, clf, metric_name, value,
                      notes=""):
        """Record a metric that is for either validation or test,
        recording the start or end of a metric, that includes a classifier
        as well as the metric and any notes that should be recorded"""
        self.check_conn()
        now = int(datetime.datetime.now().timestamp())

        try:
            self.cur.execute(
                """INSERT INTO metrics VALUES (?,?,?,?,?,?,?,?)""",
                (self.current_run, val_or_test, start_or_end, now, str(clf),
                 metric_name, value, notes))
            self.conn.commit()
        except sqlite3.Error as e:
            print(e)
            print("Values:")
            print((self.current_run, val_or_test, start_or_end, now, str(clf),
                   metric_name, value, notes))

    def save_test_predictions(self, predictions, clf, output_folder, notes=""):
        """Saves a series of predictions to a file,
        records its creation, and ends the pipeline"""
        self.check_conn()

        now = int(datetime.datetime.now().timestamp())
        fname = str(now) + ".csv"

        self.record_metric("test", "end", clf, "output file", fname, notes)
        predictions.to_csv(output_folder + "/" + fname, index=False)
        self.end_pipeline()
