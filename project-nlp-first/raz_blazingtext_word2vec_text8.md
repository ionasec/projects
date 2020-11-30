## Introduction

Word2Vec is a popular algorithm used for generating dense vector representations of words in large corpora using unsupervised learning. The resulting vectors have been shown to capture semantic relationships between the corresponding words and are used extensively for many downstream natural language processing (NLP) tasks like sentiment analysis, named entity recognition and machine translation.  

SageMaker BlazingText which provides efficient implementations of Word2Vec on

- single CPU instance
- single instance with multiple GPUs - P2 or P3 instances
- multiple CPU instances (Distributed training)

In this notebook, we demonstrate how BlazingText can be used for distributed training of word2vec using multiple CPU instances.

## Setup

Let's start by specifying:
- The S3 buckets and prefixes that you want to use for saving model data and where training data is located. These should be within the same region as the Notebook Instance, training, and hosting. If you don't specify a bucket, SageMaker SDK will create a default bucket following a pre-defined naming convention in the same region. 
- The IAM role ARN used to give SageMaker access to your data. It can be fetched using the **get_execution_role** method from sagemaker python SDK.


```python
import sagemaker
from sagemaker import get_execution_role
import boto3
import json

sess = sagemaker.Session()

role = get_execution_role()
print(role)  # This is the role that SageMaker would use to leverage AWS resources (S3, CloudWatch) on your behalf

region = boto3.Session().region_name

output_bucket = sess.default_bucket()  # Replace with your own bucket name if needed
print(output_bucket)
output_prefix = "sagemaker/DEMO-blazingtext-text8"  # Replace with the prefix under which you want to store the data if needed

data_bucket = f"jumpstart-cache-prod-{region}"  # Replace with the bucket where your data is located
data_prefix = "1p-notebooks-datasets/text8"
```

    arn:aws:iam::773208840593:role/my_AmazonSageMakerFullAccess
    sagemaker-eu-central-1-773208840593



```python

```
### Data Ingestion

BlazingText expects a single preprocessed text file with space separated tokens and each line of the file should contain a single sentence. In this example, let us train the vectors on [text8](http://mattmahoney.net/dc/textdata.html) dataset (100 MB), which is a small (already preprocessed) version of Wikipedia dump. Data is already downloaded from [matt mahoney's website](http://mattmahoney.net/dc/text8.zip), uncompressed and stored in `data_bucket`. 

```python
train_channel = f"{data_prefix}/train"

s3_train_data = f"s3://{data_bucket}/{train_channel}"
```

Next we need to setup an output location at S3, where the model artifact will be dumped. These artifacts are also the output of the algorithm's training job.


```python
s3_output_location = f"s3://{output_bucket}/{output_prefix}/output"
```

## Training Setup
Now that we are done with all the setup that is needed, we are ready to train our object detector. To begin, let us create a ``sageMaker.estimator.Estimator`` object. This estimator will launch the training job.


```python
region_name = boto3.Session().region_name
```


```python
container = sagemaker.amazon.amazon_estimator.get_image_uri(region_name, "blazingtext", "latest")
print(f"Using SageMaker BlazingText container: {container} ({region_name})")
```

    'get_image_uri' method will be deprecated in favor of 'ImageURIProvider' class in SageMaker Python SDK v2.


    Using SageMaker BlazingText container: 813361260812.dkr.ecr.eu-central-1.amazonaws.com/blazingtext:latest (eu-central-1)


## Training the BlazingText model for generating word vectors

Similar to the original implementation of [Word2Vec](https://arxiv.org/pdf/1301.3781.pdf), SageMaker BlazingText provides an efficient implementation of the continuous bag-of-words (CBOW) and skip-gram architectures using Negative Sampling, on CPUs and additionally on GPU[s]. The GPU implementation uses highly optimized CUDA kernels. To learn more, please refer to [*BlazingText: Scaling and Accelerating Word2Vec using Multiple GPUs*](https://dl.acm.org/citation.cfm?doid=3146347.3146354). BlazingText also supports learning of subword embeddings with CBOW and skip-gram modes. This enables BlazingText to generate vectors for out-of-vocabulary (OOV) words, as demonstrated in this [notebook](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/blazingtext_word2vec_subwords_text8/blazingtext_word2vec_subwords_text8.ipynb).




Besides skip-gram and CBOW, SageMaker BlazingText also supports the "Batch Skipgram" mode, which uses efficient mini-batching and matrix-matrix operations ([BLAS Level 3 routines](https://software.intel.com/en-us/mkl-developer-reference-fortran-blas-level-3-routines)). This mode enables distributed word2vec training across multiple CPU nodes, allowing almost linear scale up of word2vec computation to process hundreds of millions of words per second. Please refer to [*Parallelizing Word2Vec in Shared and Distributed Memory*](https://arxiv.org/pdf/1604.04661.pdf) to learn more.

BlazingText also supports a *supervised* mode for text classification. It extends the FastText text classifier to leverage GPU acceleration using custom CUDA kernels. The model can be trained on more than a billion words in a couple of minutes using a multi-core CPU or a GPU, while achieving performance on par with the state-of-the-art deep learning text classification algorithms. For more information, please refer to [algorithm documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html) or [the text classification notebook](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/introduction_to_amazon_algorithms/blazingtext_text_classification_dbpedia/blazingtext_text_classification_dbpedia.ipynb).

To summarize, the following modes are supported by BlazingText on different types instances:

|          Modes         	| cbow (supports subwords training) 	| skipgram (supports subwords training) 	| batch_skipgram 	| supervised |
|:----------------------:	|:----:	|:--------:	|:--------------:	| :--------------:	|
|   Single CPU instance  	|   ✔  	|     ✔    	|        ✔       	|  ✔  |
|   Single GPU instance  	|   ✔  	|     ✔    	|                	|  ✔ (Instance with 1 GPU only)  |
| Multiple CPU instances 	|      	|          	|        ✔       	|     | |

Now, let's define the resource configuration and hyperparameters to train word vectors on *text8* dataset, using "batch_skipgram" mode on two c4.2xlarge instances.



```python
bt_model = sagemaker.estimator.Estimator(
    container,
    role,
    train_instance_count=2,
    train_instance_type="ml.c4.2xlarge",
    train_volume_size=5,
    train_max_run=360000,
    input_mode="File",
    output_path=s3_output_location,
    sagemaker_session=sess,
)
```

    Parameter image_name will be renamed to image_uri in SageMaker Python SDK v2.



```python

```

    Parameter image_name will be renamed to image_uri in SageMaker Python SDK v2.


Please refer to [algorithm documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext_hyperparameters.html) for the complete list of hyperparameters.


```python
bt_model.set_hyperparameters(
    mode="batch_skipgram",
    epochs=5,
    min_count=5,
    sampling_threshold=0.0001,
    learning_rate=0.05,
    window_size=5,
    vector_dim=100,
    negative_samples=5,
    batch_size=11,  #  = (2*window_size + 1) (Preferred. Used only if mode is batch_skipgram)
    evaluation=True,  # Perform similarity evaluation on WS-353 dataset at the end of training
    subwords=False,
)  # Subword embedding learning is not supported by batch_skipgram
```

Now that the hyper-parameters are setup, let us prepare the handshake between our data channels and the algorithm. To do this, we need to create the `sagemaker.session.s3_input` objects from our data channels. These objects are then put in a simple dictionary, which the algorithm consumes.


```python
train_data = sagemaker.session.s3_input(
    s3_train_data, distribution="FullyReplicated", content_type="text/plain", s3_data_type="S3Prefix"
)
data_channels = {"train": train_data}
```

    's3_input' class will be renamed to 'TrainingInput' in SageMaker Python SDK v2.


We have our `Estimator` object, we have set the hyper-parameters for this object and we have our data channels linked with the algorithm. The only  remaining thing to do is to train the algorithm. The following command will train the algorithm. Training the algorithm involves a few steps. Firstly, the instance that we requested while creating the `Estimator` classes is provisioned and is setup with the appropriate libraries. Then, the data from our channels are downloaded into the instance. Once this is done, the training job begins. The provisioning and data downloading will take some time, depending on the size of the data. Therefore it might be a few minutes before we start getting training logs for our training jobs. The data logs will also print out `Spearman's Rho` on some pre-selected validation datasets after the training job has executed. This metric is a proxy for the quality of the algorithm. 

Once the job has finished a "Job complete" message will be printed. The trained model can be found in the S3 bucket that was setup as `output_path` in the estimator.


```python
bt_model.fit(inputs=data_channels, logs=True)
```

    2020-11-28 12:43:33 Starting - Starting the training job...
    2020-11-28 12:43:35 Starting - Launching requested ML instances......
    2020-11-28 12:44:56 Starting - Preparing the instances for training.........
    2020-11-28 12:46:21 Downloading - Downloading input data
    2020-11-28 12:46:21 Training - Downloading the training image.[34mArguments: train[0m
    [34mArguments: train[0m
    [34mFound 10.0.158.146 for host algo-1[0m
    [34mFound 10.0.151.5 for host algo-2[0m
    [35mFound 10.0.158.146 for host algo-1[0m
    [35mFound 10.0.151.5 for host algo-2[0m
    
    2020-11-28 12:46:41 Training - Training image download completed. Training in progress.[34m[11/28/2020 12:46:55 WARNING 140259373856576] Loggers have already been setup.[0m
    [34m[11/28/2020 12:46:55 WARNING 140259373856576] Loggers have already been setup.[0m
    [34m[11/28/2020 12:46:55 INFO 140259373856576] nvidia-smi took: 0.025283098220825195 secs to identify 0 gpus[0m
    [34m[11/28/2020 12:46:55 INFO 140259373856576] Running distributed CPU BlazingText training using batch_skipgram on 2 hosts.[0m
    [34m[11/28/2020 12:46:55 INFO 140259373856576] Number of hosts: 2, master IP address: 10.0.158.146, host IP address: 10.0.158.146.[0m
    [34m[11/28/2020 12:46:55 INFO 140259373856576] HTTP server started....[0m
    [34mNumber of CPU sockets found in instance is  1[0m
    [34m[11/28/2020 12:46:55 INFO 140259373856576] Processing /opt/ml/input/data/train/text8 . File size: 95.367431640625 MB[0m
    [34mWarning: Permanently added 'algo-2,10.0.151.5' (RSA) to the list of known hosts.#015[0m
    [34mprocessor name: algo-2, number of processors: 2, rank: 1[0m
    [34mprocessor name: algo-1, number of processors: 2, rank: 0[0m
    [35m[11/28/2020 12:46:55 WARNING 140443240949568] Loggers have already been setup.[0m
    [35m[11/28/2020 12:46:55 WARNING 140443240949568] Loggers have already been setup.[0m
    [35m[11/28/2020 12:46:55 INFO 140443240949568] nvidia-smi took: 0.025217056274414062 secs to identify 0 gpus[0m
    [35m[11/28/2020 12:46:55 INFO 140443240949568] Running distributed CPU BlazingText training using batch_skipgram on 2 hosts.[0m
    [35m[11/28/2020 12:46:55 INFO 140443240949568] Number of hosts: 2, master IP address: 10.0.158.146, host IP address: 10.0.151.5.[0m
    [34mRead 10M words[0m
    [34mRead 17M words[0m
    [34mNumber of words:  71290[0m
    [34mAlpha: 0.0489  Progress: 2.19%  Million Words/sec: 4.36[0m
    [34mAlpha: 0.0464  Progress: 7.23%  Million Words/sec: 4.80[0m
    [34mAlpha: 0.0439  Progress: 12.28%  Million Words/sec: 4.84[0m
    [34mAlpha: 0.0411  Progress: 17.77%  Million Words/sec: 4.85[0m
    [34mAlpha: 0.0386  Progress: 22.83%  Million Words/sec: 4.78[0m
    [34mAlpha: 0.0361  Progress: 27.83%  Million Words/sec: 4.82[0m
    [34mAlpha: 0.0336  Progress: 32.83%  Million Words/sec: 4.82[0m
    [34mAlpha: 0.0311  Progress: 37.88%  Million Words/sec: 4.85[0m
    [34mAlpha: 0.0286  Progress: 42.91%  Million Words/sec: 4.74[0m
    [34mAlpha: 0.0261  Progress: 47.92%  Million Words/sec: 4.76[0m
    [34mAlpha: 0.0233  Progress: 53.55%  Million Words/sec: 4.77[0m
    [34mAlpha: 0.0209  Progress: 58.59%  Million Words/sec: 4.79[0m
    [34mAlpha: 0.0180  Progress: 64.22%  Million Words/sec: 4.78[0m
    [34mAlpha: 0.0155  Progress: 69.26%  Million Words/sec: 4.79[0m
    [34mAlpha: 0.0130  Progress: 74.27%  Million Words/sec: 4.80[0m
    [34mAlpha: 0.0106  Progress: 79.27%  Million Words/sec: 4.80[0m
    [34mAlpha: 0.0081  Progress: 84.28%  Million Words/sec: 4.75[0m
    [34mAlpha: 0.0055  Progress: 89.32%  Million Words/sec: 4.77[0m
    [34mAlpha: 0.0027  Progress: 94.83%  Million Words/sec: 4.76[0m
    [34mAlpha: 0.0000  Progress: 99.95%  Million Words/sec: 4.68[0m
    [34mAlpha: 0.0000  Progress: 100.00%  Million Words/sec: 4.63
    [0m
    [34mTraining finished![0m
    [34mAverage throughput in Million words/sec: 4.63[0m
    [34mTotal training time in seconds: 18.37[0m
    
    2020-11-28 12:47:28 Uploading - Uploading generated training model[34mEvaluating word embeddings....[0m
    [34mVectors read from: /opt/ml/model/vectors.txt [0m
    [34m{
        "EN-WS-353-ALL.txt": {
            "not_found": 2,
            "spearmans_rho": 0.7045370515731347,
            "total_pairs": 353
        },
        "EN-WS-353-REL.txt": {
            "not_found": 1,
            "spearmans_rho": 0.6642859002083339,
            "total_pairs": 252
        },
        "EN-WS-353-SIM.txt": {
            "not_found": 1,
            "spearmans_rho": 0.7317117920029901,
            "total_pairs": 203
        },
        "mean_rho": 0.7001782479281529[0m
    [34m}[0m
    [34m[11/28/2020 12:47:25 INFO 140259373856576] #mean_rho: 0.7001782479281529[0m
    [35m[11/28/2020 12:47:25 INFO 140443240949568] Master host is not alive. Training might have finished. Shutting down.... Check the logs for algo-1 machine.[0m
    
    2020-11-28 12:47:55 Completed - Training job completed
    Training seconds: 220
    Billable seconds: 220


## Hosting / Inference
Once the training is done, we can deploy the trained model as an Amazon SageMaker real-time hosted endpoint. This will allow us to make predictions (or inference) from the model. Note that we don't have to host on the same type of instance that we used to train. Because instance endpoints will be up and running for long, it's advisable to choose a cheaper instance for inference.


```python
bt_endpoint = bt_model.deploy(initial_instance_count=1, instance_type="ml.m4.xlarge")
```

    Parameter image will be renamed to image_uri in SageMaker Python SDK v2.


    -----------------!

### Getting vector representations for words

#### Use JSON format for inference
The payload should contain a list of words with the key as "**instances**". BlazingText supports content-type `application/json`.


```python
words = ["muie"]

payload = {"instances": words}

response = bt_endpoint.predict(
    json.dumps(payload), initial_args={"ContentType": "application/json", "Accept": "application/json"}
)

vecs = json.loads(response)
print(vecs)
```

    [{'vector': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], 'word': 'muie'}]


As expected, we get an n-dimensional vector (where n is vector_dim as specified in hyperparameters) for each of the words. If the word is not there in the training dataset, the model will return a vector of zeros.

### Evaluation

Let us now download the word vectors learned by our model and visualize them using a [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) plot.


```python
s3 = boto3.resource("s3")

key = bt_model.model_data[bt_model.model_data.find("/", 5) + 1 :]
s3.Bucket(output_bucket).download_file(key, "model.tar.gz")
```

Uncompress `model.tar.gz` to get `vectors.txt`


```python
!tar -xvzf model.tar.gz
```

    eval.json
    vectors.txt
    vectors.bin


If you set "evaluation" as "true" in the hyperparameters, then "eval.json" will be there in the model artifacts.

The quality of trained model is evaluated on word similarity task. We use [WS-353](http://alfonseca.org/eng/research/wordsim353.html), which is one of the most popular test datasets used for this purpose. It contains word pairs together with human-assigned similarity judgments.

The word representations are evaluated by ranking the pairs according to their cosine similarities, and measuring the Spearmans rank correlation coefficient with the human judgments.

Let's look at the evaluation scores which are there in eval.json. For embeddings trained on the text8 dataset, scores above 0.65 are pretty good.


```python
!cat eval.json
```

    {
        "EN-WS-353-ALL.txt": {
            "not_found": 2,
            "spearmans_rho": 0.7045370515731347,
            "total_pairs": 353
        },
        "EN-WS-353-REL.txt": {
            "not_found": 1,
            "spearmans_rho": 0.6642859002083339,
            "total_pairs": 252
        },
        "EN-WS-353-SIM.txt": {
            "not_found": 1,
            "spearmans_rho": 0.7317117920029901,
            "total_pairs": 203
        },
        "mean_rho": 0.7001782479281529
    }

Now, let us do a 2D visualization of the word vectors


```python
import numpy as np
from sklearn.preprocessing import normalize

# Read the 400 most frequent word vectors. The vectors in the file are in descending order of frequency.
start = 200
stop = 1000
num_points = stop-start


first_line = True
index_to_word = []
with open("vectors.txt", "r") as f:
    for line_num, line in enumerate(f):
        if first_line:
            dim = int(line.strip().split()[1])
            word_vecs = np.zeros((stop-start, dim), dtype=float)
            first_line = False
            continue
        if line_num<start:
            continue
            
        line = line.strip()
        word = line.split()[0]
       # print(word)
        
        vec = word_vecs[line_num-start - 1]
        
        for index, vec_val in enumerate(line.split()[1:]):
            vec[index] = float(vec_val)
        index_to_word.append(word)
        if line_num >= stop:
            break
word_vecs = normalize(word_vecs, copy=False, return_norm=False)
```


```python
from sklearn.manifold import TSNE

tsne = TSNE(perplexity=40, n_components=2, init="pca", n_iter=10000)
two_d_embeddings = tsne.fit_transform(word_vecs[:num_points])
labels = index_to_word[:num_points]
```


```python
from matplotlib import pylab
%matplotlib inline

def plot(embeddings, labels):
    pylab.figure(figsize=(20, 20))
    for i, label in enumerate(labels):
        x, y = embeddings[i, :]
        pylab.scatter(x, y)
        pylab.annotate(
            label, xy=(x, y), xytext=(5, 2), textcoords="offset points", ha="right", va="bottom"
        )
    pylab.show()


plot(two_d_embeddings, labels)
```


![png](output_42_0.png)


Running the code above might generate a plot like the one below. t-SNE and Word2Vec are stochastic, so although when you run the code the plot won’t look exactly like this, you can still see clusters of similar words such as below where 'british', 'american', 'french', 'english' are near the bottom-left, and 'military', 'army' and 'forces' are all together near the bottom.

![tsne plot of embeddings](./tsne.png)

### Stop / Close the Endpoint (Optional)
Finally, we should delete the endpoint before we close the notebook.


```python
sess.delete_endpoint(bt_endpoint.endpoint)
```
