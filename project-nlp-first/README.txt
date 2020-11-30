1) Train word2vec on a corpus and display the 2d similary map.
2) Perform name entity recognition of a test paragraph
3) Setup a voice to text interactive pipeline 
4) Setup a sentiment analysis of the input text pipeline


Unsupervised Learning of Word2Vec using BlazingTest
Explaining of Continous Bag of Words and Skipgram architectures for word embeddings
https://towardsdatascience.com/nlp-101-word2vec-skip-gram-and-cbow-93512ee24314#:~:text=In%20the%20CBOW%20model%2C%20the,used%20to%20predict%20the%20context%20.
Negative Sampling and other acceleration techiques
https://towardsdatascience.com/nlp-101-negative-sampling-and-glove-936c88f3bc68
N-gram learning to encode subwords and learn out-of-vocabulary Words
https://towardsdatascience.com/introduction-to-natural-language-processing-for-text-df845750fb63
https://docs.aws.amazon.com/sagemaker/latest/dg/blazingtext.html
*Batch_skipgram can do multicpu for Unsupervised
https://t-redactyl.io/blog/2020/09/training-and-evaluating-a-word2vec-model-using-blazingtext-in-sagemaker.html


Suprervized Learning with BlazingText - multi class and multi - labels
It extends the FastText text classifier to leverage GPU acceleration using custom CUDA kernels. The model can be trained on more than a billion words in a couple of minutes using a multi-core CPU or a GPU, while achieving performance on par with the state-of-the-art deep learning text classification algorithms.
*No multi instance only multi CPU/GPU for superivzed


Latent Dirichlet Allocation (LDA) Algorithm - Unsupervised topic categorization / potential not align with human defined topics. 
https://docs.aws.amazon.com/sagemaker/latest/dg/lda.html
Each observation is a doucment. Features are count of each word. And categories are the topics. Each topic is are learned as a probability distrubiton over the features. Each document is a mixture of probabilities. 
Input / protobuf (desnse and sparse) stream and CSV format - matrix - number of records * vocabulary size. 
Bag of words, the order does not matter. Generative model. 
https://raz-sagemaker-notebook-frankfurt.notebook.eu-central-1.sagemaker.aws/lab

Bag-of-words or TFIDF encoding of documents 
https://towardsdatascience.com/introduction-to-natural-language-processing-for-text-df845750fb63


Neural Topic Model (NTM) Algorithm - Unsupervised topic categorization
Final similar documents. Find similar buyer groups etc. Input to a doucment classifier. 
With Aux supply the vocabulary. User case see the top words for each topic cluster. Compute the WETC score - simlary of top words 




Blazing Text / word2vec
https://t-redactyl.github.io/blog/2020/09/training-and-evaluating-a-word2vec-model-using-blazingtext-in-sagemaker.html
https://towardsdatascience.com/training-word-embeddings-on-aws-sagemaker-using-blazingtext-93d0a0838212
Analyze content with Amazon Comprehend and Amazon SageMaker notebooks
https://aws.amazon.com/blogs/machine-learning/analyze-content-with-amazon-comprehend-and-amazon-sagemaker-notebooks/
