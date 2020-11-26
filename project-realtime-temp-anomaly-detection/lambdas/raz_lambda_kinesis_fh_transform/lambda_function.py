from __future__ import print_function

import base64
import json
from lambda_invoke_sagemaker import predict_anomaly

print('Loading function')


def lambda_handler(event, context):
  output = []
  
  for record in event['records']:
      payload = base64.b64decode(record['data']).decode('utf-8')
      print("Decoded payload: " + payload)
      
    #  prediction = predict_anomaly(payload,context=0)
      payload = payload + '\n'
    
      bpayload = base64.b64encode(payload.encode('utf-8'))
      print("Encoded payload: " + str(bpayload))
  
  
      # Do custom processing on the payload here
      output_record = {
         'recordId': record['recordId'],
         'result': 'Ok',
         'data': bpayload
      }
      output.append(output_record)

  print('Successfully processed {} records.'.format(len(event['records'])))
  return {'records': output}