import logging
import azure.functions as func
import os
import numpy as np
import pandas as pd
import torch
from torch import nn
from narx import NARX_4


from sysidentpy.metrics import mean_squared_error, root_relative_squared_error
from sysidentpy.neural_network import NARXNN
from sysidentpy.basis_function._basis_function import Polynomial, Fourier
from sysidentpy.utils.save_load import load_model

from sqlalchemy import create_engine

from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()

username = os.getenv("DB_USER")
password = os.getenv("DB_PASSWORD")
db_connection = "final-project.postgres.database.azure.com:5432/postgres?sslmode=require"

CONNECTION = f"postgresql://{username}:{password}@{db_connection}"

# Import the trained neural network 
my_model = load_model(file_name='narx_net.syspy')

# Factors for the ARX model
a_1 = 0.76185
c_1 = -80.3036
b_3 = 0.86892
b_4 = 0.78741
b_2 = 3.4706
b_1 = 2.9112

# ARX model that was obtained through sklearn toolbox
def my_arx_model(x1,x2,y):
    arx = np.zeros(x1.size)
    for t in range(17, x1.size):
        arx[t] = (c_1 + a_1*y[t-1]+b_1*x1[t-1]+b_2*x2[t-1]+b_3*x1[t-17]+b_4*x2[t-17])/10            
    return arx 


app = func.FunctionApp()

# Periodic function that retrieves data from a database and runs predictions based on 2 models
# Pushes the results into a new database
@app.schedule(schedule="0 */1 * * * *", arg_name="myTimer", run_on_startup=True,
              use_monitor=False) 
def test_model(myTimer: func.TimerRequest) -> None:
    import psycopg2
    if myTimer.past_due:
        logging.info('The timer is past due!')  

    # Connect to the database to retrieve data
    conn = psycopg2.connect(CONNECTION) 
    data = pd.read_sql("SELECT * from pump_data;", conn)
    logging.info('data got retrieved from database')
    try:
        curs = conn.cursor()

    # Handle exception if connection wasn't possible
    except psycopg2.InterfaceError as e:
            logging.info('{} - connection will be reset'.format(e))  
            # Close old connection
            if conn:
                if curs:
                        curs.close()
                conn.close()
            conn = None
            curs = None

            # Reconnect
            conn = psycopg2.connect(CONNECTION)
            curs = conn.cursor()
    # Clear the output database because to this moment appending makes no sense as always the same data is used to run on the model
            
    curs.execute('TRUNCATE TABLE prediction_results')
    count = curs.rowcount
    print(f"Counter Variable {count}")
    conn.commit()
    curs.close()
    conn.close()
    logging.info('Python timer trigger function executed.')

    # Set the time column of the retrieved data to be the index
    data['time'] = pd.to_datetime(data['time']) # convert column to datetime object
    data.set_index('time', inplace=True) # set column 'date' to index

    # Get data for pump 4
    data_10s = data[:]['2023-02-15 11:00:00':'2023-02-15 13:00:00'].resample("10s").mean()
    
    # visualisation_frame = data_4_10s.drop("height", 'pump1_rpm', 'pump1_power', 'pump4_power')
    #logging.info(data_10s[0:200])
    x_compare_4 = data_10s['pump4_rpm'].values.reshape(-1,1)
    x_compare_1 = data_10s['pump1_rpm'].values.reshape(-1,1)
    y_compare = data_10s['outflow'].values.reshape(-1,1)
    timestamps = data_10s.index

    # Check the performance of the ARX model and safe the data to later push it to the database
    yhat = my_arx_model(x_compare_1,x_compare_4,y_compare)

    

    # Normalize the data to run a prediction on the neural network
    data_10s = data_10s.drop(labels=['height', 'pump1_power','pump4_power', 'pump1_rpm'], axis=1)
    data_10s = data_10s.reset_index(drop=True)   
    scaled_data_10s = scaler.fit_transform(data_10s)

    compare_data = pd.DataFrame(scaled_data_10s,columns=data_10s.columns, index=data_10s.index)
    x_compare_4_scaled = compare_data['pump4_rpm'].values.reshape(-1,1)
    y_compare_scaled = compare_data['outflow'].values.reshape(-1,1)

    # Load the trained neural network and run a prediction
    my_model = load_model(file_name='narx_net.syspy')
    
    yhat_compare = my_model.predict(X=x_compare_4_scaled,y=y_compare_scaled, forecast_horizon=400)
  
    # Calculate relative error and log it to the console
    rrse = root_relative_squared_error(y_compare_scaled, yhat_compare)
    logging.info(f"The RRSE is equal to {rrse}")

    # Create new dataframe to be stored in the database to then be visualized by grafana
    result = pd.DataFrame({'time':timestamps.ravel(), 'pump1_rpm': x_compare_1.ravel(), 'pump4_rpm':x_compare_4.ravel(), 
                           'outflow':y_compare.ravel(), 'prediction': yhat.ravel(), 'pump4_rpm_scaled': x_compare_4_scaled.ravel(), 
                           'outflow_scaled': y_compare_scaled.ravel(), 'nn_predict': yhat_compare.ravel()})
    
    # Set the time column to the index of the dataframe
    result['time'] = pd.to_datetime(result['time'])
    result.set_index('time', inplace=True)
    # Push new data into the Database
    db = create_engine(CONNECTION)

    a = result.to_sql('prediction_results', con=db, if_exists='append', index=True, index_label='time')


    

