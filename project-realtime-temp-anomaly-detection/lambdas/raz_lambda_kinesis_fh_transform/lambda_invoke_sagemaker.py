import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = 'linear-learner-2020-11-24-18-50-45-646'
runtime= boto3.client('runtime.sagemaker')


def predict_anomaly(event, context):
    #load to json
    data = json.loads(event)
    observation = str(data['PRESSURE'])+","+str(data['TEMPERATURE'])+","+str(data['HUMIDITY'])

    data = {"data":observation}
    payload = data['data']
   
    response = runtime.invoke_endpoint(EndpointName=ENDPOINT_NAME,
                                       ContentType='text/csv',
                                       Body=payload)
  
    result = json.loads(response['Body'].read().decode())
    print(result)
    print((result['predictions'])[0]['predicted_label'])
    
    return result