# Licensed under the MIT license.

from azureml.core import Model,Workspace
import joblib
import sys
sys.path.append("..")
import pandas as pd
from inference_schema.parameter_types.pandas_parameter_type import PandasParameterType
from azureml.core.authentication import MsiAuthentication
import json
import ast

def init():
    global ws
    subscription_id = 'SUBSCRIPTION_ID'
    resource_group = 'RESOURCE_GROUP' 
    workspace_name = 'AML_WORKSPACE_NAME'

    try:
        msi_auth = MsiAuthentication()
        print("MSI is successful")
    except:
        print("Unable to Authenticate with MSI")

    ws = Workspace(subscription_id=subscription_id,
                    resource_group=resource_group,
                    workspace_name=workspace_name,
                    auth=msi_auth)


    print("Found workspace {} at location {}".format(ws.name, ws.location))

def run(raw_data):
    Inputs = pd.DataFrame(ast.literal_eval(json.loads(raw_data)['Inputs']))

    timestamp_column= 'WeekStarting'
    Inputs[timestamp_column]=pd.to_datetime(Inputs[timestamp_column])

    timeseries_id_columns= [ 'Store', 'Brand']
    data = Inputs \
            .set_index(timestamp_column) \
            .sort_index(ascending=True)

    #Prepare loading model from Azure ML, get the latest model by default
    model_name="prs_"+str(data['Store'].iloc[0])+"_"+str(data['Brand'].iloc[0])
    model = Model(ws, model_name)
    model.download(exist_ok =True)
    forecaster = joblib.load(model_name)

    #   Get predictions 
    #This is to append the store and brand column to the result
    ts_id_dict = {id_col: str(data[id_col].iloc[0]) for id_col in timeseries_id_columns}
    forecasts=forecaster.forecast(data)
    prediction_df = forecasts.to_frame(name='Prediction')
    prediction_df =prediction_df.reset_index().assign(**ts_id_dict)
  

    return prediction_df.to_json()
