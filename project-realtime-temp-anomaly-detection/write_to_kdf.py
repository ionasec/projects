 
import json
import boto3
import random
import datetime
import time


kinesis = boto3.client('kinesis')


def getReferrer():
    data = {}
    now = datetime.datetime.now()
    str_now = now.isoformat()
    data['EVENT_TIME'] = str_now
    data['TICKER'] = random.choice(['AAPL', 'AMZN', 'MSFT', 'INTC', 'TBV'])
    price = random.random() * 100
    data['PRICE'] = round(price, 2)
    return data




while 1:
    data = json.dumps(getReferrer())+"\n"
    print(data)
    time.sleep(3)
    kinesis.put_record(
                StreamName="raz-test-kinesis-fh-newdev",
                Data=data,
                PartitionKey="partitionkey-NEW1")
 
