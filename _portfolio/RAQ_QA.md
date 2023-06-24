---
title: "Q&A on Custom Dataset Using Retrieval-Augmented Generation (RAG) with GPT-J Embeddings"
excerpt: <br/><img src='/images/thumb6.png' width =350>
collection: portfolio

---

# Introduction


I have been working on an exciting project using SageMaker for Natural Language Processing (NLP). With SageMaker, I built a powerful text embedding model that can understand and analyze textual data from a custom dateset. By leveraging state-of-the-art RAG algorithm and the good old KNN, I developed a system capable of generating meaningful and factually correct answers from a given textual dataset. 


```python
!pip install --upgrade sagemaker --quiet
!pip install ipywidgets==7.0.0 --quiet
```


```python
import time
import sagemaker
import boto3
import json
from sagemaker.session import Session as SM_Session
from sagemaker.model import Model as SM_Model
from sagemaker import (
    image_uris,
    model_uris,
    script_uris,
    hyperparameters
)
from sagemaker.predictor import Predictor as SM_Predictor
from sagemaker.utils import name_from_base as sm_name_from_base

sm_session = SM_Session()
sm_role = sm_session.get_caller_identity_arn()
sm_region = boto3.Session().region_name
sm_session = sagemaker.Session()
sm_version = "*"


```


## Document Retrieval and LLM Integration

In this project, the primary objective is to leverage document embeddings to retrieve the most relevant documents from our extensive knowledge library. These documents will then be combined with the prompt I provide to the Language Model (LLM) for further analysis and processing.

To achieve this, I have outlined the following steps:

## 1. Generate Document Embeddings

I will utilize the powerful GPT-J-6B embedding model to generate embeddings for each document in our knowledge library. These embeddings capture the semantic meaning and contextual information of the documents.

## 2. Identify Top K Relevant Documents

Given a user query, I will follow these steps to identify the top K most relevant documents:

- I will generate an embedding for the user query using the same GPT-J-6B embedding model.
- Utilizing the SageMaker KNN algorithm, I will search for the top K most relevant documents based on the embedding space. This algorithm allows me to efficiently and accurately retrieve documents that closely match the query.
- Using the indexes obtained from the KNN algorithm, I will retrieve the corresponding documents.

## 3. Combine Retrieved Documents with Prompt and Question

To provide a comprehensive input to the LLM, I will merge the retrieved documents with the prompt and question. This consolidated input will be utilized to further analyze and process the information using the LLM. It is crucial to ensure that the combined document and text are of an appropriate size, containing enough information to answer the question while adhering to the maximum sequence length of 1024 tokens required by the LLM.

By following these steps, I aim to enhance the information retrieval process by leveraging document embeddings and harnessing the capabilities of the GPT-J-6B model. This approach empowers me to extract relevant information and create a more comprehensive input for the LLM, enabling accurate analysis and insightful responses.

Please note that the retrieved documents need to strike a balance between containing sufficient information and fitting within the constraints of the LLM's maximum sequence length of 1024 tokens. This ensures efficient processing and optimal utilization of the LLM's capabilities.


### Deploying the model endpoint for GPT-J-6B embedding model


```python
import random
import string
import sagemaker
import boto3
from sagemaker.model import Model as SM_Model
from sagemaker import image_uris, model_uris
from sagemaker.session import Session as SM_Session
from sagemaker.predictor import Predictor as SM_Predictor

def generate_random_string(length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for i in range(length))

model_id, model_version = "text-embedding-gpt-j-6b", "*"
endpoint_name = generate_random_string(10)

inference_instance_type = "ml.g5.24xlarge"

deploy_image_uri = image_uris.retrieve(
    region=boto3.Session().region_name,
    framework="pytorch",
    image_scope="inference",
    model_id=model_id,
    model_version=model_version,
    instance_type=inference_instance_type,
)

model_uri = model_uris.retrieve(
    model_id=model_id,
    model_version=model_version,
    model_scope="inference"
)

model = SM_Model(
    image_uri=deploy_image_uri,
    model_data=model_uri,
    role=sm_role,
    name=endpoint_name,
    env={"TS_DEFAULT_WORKERS_PER_MODEL": "1"}
)

model_predictor = model.deploy(
    initial_instance_count=1,
    instance_type=inference_instance_type,
    endpoint_name=endpoint_name,
)



```


```python
def query_endpoint(encoded_json, endpoint, content_type="application/json"):
    client = boto3.client("runtime.sagemaker")
    response = client.invoke_endpoint(
        EndpointName=endpoint, ContentType=content_type, Body=encoded_json
    )
    return response

def parse_query_response(response):
    predictions = json.loads(response["Body"].read())
    embeddings = predictions["embedding"]
    return embeddings

def build_embedding_table(df, endpoint, column, batch_size=10):
    embeddings = []
    for idx in tqdm(range(0, df.shape[0], batch_size)):
        content = df.loc[idx : (idx + batch_size - 1)][column].tolist()
        payload = {"text_inputs": content}
        response = query_endpoint(
            json.dumps(payload).encode("utf-8"), endpoint
        )
        generated_embeddings = parse_query_response(response)
        embeddings.extend(generated_embeddings)
    embeddings_df = pd.DataFrame(embeddings)
    return embeddings_df

endpoint_name = generate_random_string(10)
embeddings_df = build_embedding_table(data_frame, endpoint_name, column_name)

```

### Indexing the embedding knowledge library using SageMaker KNN algorithm

## 1. Training Job for Embedding Indexing

I start the KNN by initiating a training job that creates an index for the embedding knowledge data. To accomplish this, I employ the powerful [Faiss](https://github.com/facebookresearch/faiss) algorithm, which efficiently indexes the data for similarity search.

## 2. Endpoint for Nearest Document Retrieval

Once the indexing is complete, I create the endpoint. This endpoint handles the task of accepting query embeddings and promptly returning the top K nearest indexes of the relevant documents. 

**Note:** During the KNN training job,  matrix of features represented by an N by P structure. Here, N denotes the number of documents in the knowledge library, P represents the embedding dimension, and each row elegantly corresponds to the embedding of a document. 



```python
import numpy as np
import os
import io
from sagemaker.session import Session as SM_Session
import sagemaker.amazon.common as smac

train_features = np.array(df_knowledge_embed)

# Providing each answer embedding label
train_labels = np.array([i for i in range(len(train_features))])

print("train_features shape =", train_features.shape)
print("train_labels shape =", train_labels.shape)

buf = io.BytesIO()
smac.write_numpy_to_dense_tensor(buf, train_features, train_labels)
buf.seek(0)

bucket_name = SM_Session().default_bucket()  # modify to your bucket name
prefix = "RAGDatabase"
key = "Amazon-SageMaker-RAG"

s3_client = boto3.client("s3")
s3_client.upload_fileobj(buf, bucket_name, os.path.join(prefix, "train", key))
s3_train_data = f"s3://{bucket_name}/{prefix}/train/{key}"
print(f"Uploaded training data location: {s3_train_data}")

TOP_K = 5

```

```python
from sagemaker.session import Session as SM_Session
from sagemaker.amazon.amazon_estimator import get_image_uri
from sagemaker.estimator import Estimator as SM_Estimator


def create_estimator(train_data, hyperparameters, output_path):
    """
    Create an Estimator from the given hyperparameters, fit to training data,
    and return a deployed predictor.
    """
    # Set up the estimator
    knn = SM_Estimator(
        get_image_uri(boto3.Session().region_name, "knn"),
        role,
        instance_count=1,
        instance_type="ml.m5.2xlarge",
        output_path=output_path,
        sagemaker_session=session
    )
    knn.set_hyperparameters(**hyperparameters)

    # Train a model. fit_input contains the locations of the train data
    fit_input = {"train": train_data}
    knn.fit(fit_input)
    return knn

hyperparameters = {
    "feature_dim": train_features.shape[1],
    "k": TOP_K,
    "sample_size": train_features.shape[0],
    "predictor_type": "classifier"
}
output_path = f"s3://{bucket_name}/{prefix}/default_example/output"
knn_estimator = create_estimator(s3_train_data, hyperparameters, output_path)


```

Deploying the KNN endpoint for retrieving indexes of top K most relevant docuemnts.


```python
from sagemaker.serializers import CSVSerializer
from sagemaker.deserializers import JSONDeserializer


def predictor_from_estimator(estimator, instance_type, endpoint_name=None):
    predictor = estimator.deploy(
        initial_instance_count=1,
        instance_type=instance_type,
        endpoint_name=endpoint_name
    )
    predictor.serializer = CSVSerializer()
    predictor.deserializer = JSONDeserializer()
    return predictor


instance_type = "ml.m4.xlarge"
endpoint_name = name_from_base(f"jumpstart-example-ragknn")

knn_predictor = predictor_from_estimator(knn_estimator, instance_type, endpoint_name=endpoint_name)

```

### 4.4 Retrieve the most relevant documents

Given the embedding of a query, the endpoint is queried to obtain the indexes of the top K most relevant documents. These indexes will then be used to retrieve the corresponding textual documents.

The textual documents are retrieved while ensuring that the combined length does not exceed MAX_SECTION_LEN. This step is crucial to ensure that the context sent into the prompt contains a substantial amount of information while staying within the capacity of the model. 

```python

MAX_SECTION_LEN = 500
SEPARATOR = "\n* "

def construct_context(context_predictions, data_frame) -> str:
    selected_sections = []
    total_length = 0

    for index in context_predictions:
        document_section = data_frame.loc[index]
        section_length = len(document_section) + 2
        if total_length + section_length > MAX_SECTION_LEN:
            break

        selected_sections.append(SEPARATOR + document_section.replace("\n", " "))
        total_length += section_length

    concatenated_doc = "".join(selected_sections)
    print(
        f"With maximum sequence length {MAX_SECTION_LEN}, selected top {len(selected_sections)} document sections: {concatenated_doc}"
    )

    return concatenated_doc

```


```python

start_time = time.time()

query_response = query_endpoint_with_json_payload(
    question, endpoint_name, content_type="application/x-text"
)
question_embedding = parse_response_multiple_texts(query_response)

# Getting the most relevant context using KNN
context_predictions = knn_predictor.predict(
    np.array(question_embedding),
    initial_args={"ContentType": "text/csv", "Accept": "application/json; verbose=true"},
)["predictions"][0]["labels"]

context_embed_retrieve = construct_context(context_predictions, df_knowledge["Answer"])

elapsed_time = time.time() - start_time

print(
    f"\nElapsed time for computing the embedding of a query and retrieving the top K most relevant documents: {elapsed_time} seconds.\n"
)


```

### Combining the retrieved documents, prompt, and question to query the LLM


```python

for model_id in _MODEL_CONFIG_:
    endpoint_name = _MODEL_CONFIG_[model_id]["endpoint_name"]

    model_prompt = _MODEL_CONFIG_[model_id]["prompt"]

    text_input = model_prompt.replace("{context}", context_embed_retrieve)
    text_input = text_input.replace("{question}", question)

    payload = {"text_inputs": text_input, **parameters}

    query_response = query_endpoint_with_json_payload(
        json.dumps(payload).encode("utf-8"), endpoint_name=endpoint_name
    )
    generated_texts = _MODEL_CONFIG_[model_id]["parse_function"](query_response)
    print(f"For model: {model_id}, the generated output is: {generated_texts[0]}\n")


```
