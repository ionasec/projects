 
import json
import boto3
import random
import datetime
import time

DELIVERY_STEAM = "raz-iot-kinesis-fh-deliverystream-dev"
PARTION_KEY = "partitionkey-iot-1"
TIMEOUT = 1

kinesis = boto3.client('kinesis')


def getData():
    data = {}
    now = datetime.datetime.now()
    str_now = now.isoformat()
    data['EVENT_TIME'] = str_now
    data['TICKER'] = random.choice(['AAPL', 'AMZN', 'MSFT', 'INTC', 'TBV'])
    price = random.random() * 100
    data['PRICE'] = round(price, 2)
    return data




while 1:
    data = json.dumps(getData())+"\n"
    print(data)
    time.sleep(TIMEOUT)
    kinesis.put_record(
                StreamName=DELIVERY_STEAM,
                Data=data,
                PartitionKey=PARTION_KEY)
 
