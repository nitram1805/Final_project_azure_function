import logging
import azure.functions as func
import os

import pandas as pd

username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
db_connection = "final-project.postgres.database.azure.com:5432/postgres?sslmode=require"

CONNECTION = f"postgres://{username}:{password}@{db_connection}"


app = func.FunctionApp()

@app.schedule(schedule="0 */1 * * * *", arg_name="myTimer", run_on_startup=True,
              use_monitor=False) 
def test_model(myTimer: func.TimerRequest) -> None:
    import psycopg2
    if myTimer.past_due:
        logging.info('The timer is past due!')
        
        
        

    with psycopg2.connect(CONNECTION) as conn:
            data = pd.read_sql("SELECT * from pump_data;", conn)
            logging.info('data got retrieved from database')
            logging.info(data[0:200])
    logging.info('Python timer trigger function executed.')
 