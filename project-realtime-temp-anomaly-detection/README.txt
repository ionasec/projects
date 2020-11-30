Functional:
- Read real-time temperature measurements from Raspberry Pie
- Push data using KDS / KDF in CSV format to AWS
- Use KDA analytics to detectect anomalies - https://docs.aws.amazon.com/kinesisanalytics/latest/dev/app-anomaly-detection.html,
https://docs.aws.amazon.com/kinesisanalytics/latest/sqlref/sqlrf-random-cut-forest-with-explanation.html
https://medium.com/a-tale-of-2-from-data-to-information/how-to-build-an-event-pipeline-part-2-transforming-records-using-lambda-functions-d68cf3e879ed
https://stackoverflow.com/questions/50352545/kinesis-firehose-lambda-transformation


- Send SMS / EMAIL if anomaly detection

- Write data to S3 Buckeet - CSV + Anomaly yes / now

- Use GLUE to create schema

- Use SageMaker to train anaomaly detection


- Compare Comparison for Binary Classification SAgeMaker and Anomaly Detection with GT
Non-Functiona:
- Encryption in transit and at-rest
- Low latency to SMS receiving
- Lowest cost per 1h of operations



*CloudFormation Code to build AWS Backend for KDS / KDF  - Use

*Python Code to read Pie 
*Python Code to send reading to KDS
