import json
import boto3
import random
import datetime
import time
import bme280

DELIVERY_STEAM = "raz-iot-kinesis-fh-dev"
PARTION_KEY = "partitionkey-iot-1"
TIMEOUT = 1

kinesis = boto3.client('kinesis')


def getData():

    temperature,pressure,humidity = bme280.readBME280All()

    data = {}
    now = datetime.datetime.now()
    str_now = now.isoformat()
    data['EVENT_TIME'] = str_now
    data['TEMPERATURE'] = temperature
    data['PRESSURE'] = pressure
    data['HUMIDITY'] = humidity

    return data




while 1:
    data = json.dumps(getData())+"\n"
    print(data)
    time.sleep(TIMEOUT)
    kinesis.put_record(
                StreamName=DELIVERY_STEAM,
                Data=data,
                PartitionKey=PARTION_KEY)
