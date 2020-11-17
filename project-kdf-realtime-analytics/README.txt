Step 1: Create a kinesis fh with a cloud formation template using the AWS CLI

aws cloudformation create-stack --stack-name raz-myteststack --template-body cloudformation-kinesis-fh-delivery-stream.json


Step 2: Setup a KDF and send dummy data to it from with a python SDK and a script every 3 seconds


Prerequisits
https://docs.aws.amazon.com/cli/latest/userguide/install-cliv2-windows.html#cliv2-windows-install

https://docs.aws.amazon.com/general/latest/gr/aws-access-keys-best-practices.html



https://docs.aws.amazon.com/firehose/latest/dev/what-is-this-service.html

Amazon Kinesis Data Firehose is a fully managed service for delivering real-time streaming data to destinations such as Amazon Simple Storage Service (Amazon S3), Amazon Redshift, Amazon Elasticsearch Service (Amazon ES), Splunk, and any custom HTTP endpoint or HTTP endpoints owned by supported third-party service providers, including Datadog, MongoDB, and New Relic. Kinesis Data Firehose is part of the Kinesis streaming data platform, along with Kinesis Data Streams, Kinesis Video Streams, and Amazon Kinesis Data Analytics. With Kinesis Data Firehose, you don't need to write applications or manage resources. You configure your data producers to send data to Kinesis Data Firehose, and it automatically delivers the data to the destination that you specified. You can also configure Kinesis Data Firehose to transform your data before delivering it.

send data
1) Use the Kinesis Data Firehose PutRecord() or PutRecordBatch() API to send source records to the delivery stream. Learn more 
2) The Amazon Kinesis Agent is a stand-alone Java software application that offers an easy way to collect and send source records to Kinesis Data Firehose
3) Create AWS IoT rules that send data from MQTT messages
4) CloudWatch Logs Use subscription filters to deliver a real-time stream of log events
5) CloudWatch Events Create rules to indicate which events are of interest to your application and what automated action to take when a rule matches an event.

record
The data of interest that your data producer sends to a Kinesis Data Firehose delivery stream. A record can be as large as 1,000 KB.

buffer size and buffer interval
Kinesis Data Firehose buffers incoming streaming data to a certain size or for a certain period of time before delivering it to destinations. Buffer Size is in MBs and Buffer Interval is in seconds.


destination
For Amazon Redshift destinations, streaming data is delivered to your S3 bucket first. Kinesis Data Firehose then issues an Amazon Redshift COPY command to load data from your S3 bucket to your Amazon Redshift cluster.If data transformation is enabled, you can optionally back up source data to another Amazon S3 bucket.

For Amazon ES destinations, streaming data is delivered to your Amazon ES cluster, and it can optionally be backed up to your S3 bucket concurrently.

For Splunk destinations, streaming data is delivered to Splunk, and it can optionally be backed up to your S3 bucket concurrently.

Put data
Single Write Operations Using PutRecord

Batch Write Operations Using PutRecordBatch (up to 500 records)
https://docs.aws.amazon.com/firehose/latest/APIReference/API_Operations.html

Perforamnce
By default, each delivery stream can take in up to 2,000 transactions per second, 5,000 records per second, or 5 MB per second.
The PutRecord operation returns a RecordId, which is a unique string assigned to each record. Producer applications can use this ID for purposes such as auditability and investigation.

aws firehose put-record --delivery-stream-name mystream --record="{\"Data\":\"1\"}"