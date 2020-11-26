import os
import io
import boto3
import json
import csv

# grab environment variables
ENDPOINT_NAME = 'linear-learner-2020-11-24-18-50-45-646'
runtime= boto3.client('runtime.sagemaker')


#PRESSURE	TEMPERATURE	HUMIDITY
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
       
    return result['predictions'][0]['predicted_label']

def main():
    #{"PRESSURE":990.5552533448181,"EVENT_TIME":"2020-11-22T19:29:09.382843","TEMPERATURE":21.64,"HUMIDITY":41.51219248842094,"ANOMALY_SCORE":0.9324740502040736}
    data = '{"PRESSURE":990.5552533448181,"EVENT_TIME":"2020-11-22T19:29:09.382843","TEMPERATURE":22.64,"HUMIDITY":41.51219248842094,"ANOMALY_SCORE":0.9324740502040736}'
    result = predict_anomaly(data, context=0)
    temp = json.loads(data)
    temp["PREDICTION"] = result
    data = json.dumps(temp)
    print("start")
    print(data)
    print("end")

if __name__ == "__main__":
    main()



