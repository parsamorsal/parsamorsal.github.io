---
title: "Custom Named Entity Recognition (NER) Using spaCy"
excerpt: <br/><img src='/images/thumb4.png' width =350>
collection: portfolio
---


## Name Entity Recognization (NER)

Named Entity Recognition (NER) is a subtask of Natural Language Processing (NLP) that aims to identify and classify named entities in unstructured text, such as people, organizations, locations, dates, and more. NER has a wide range of applications across various domains, including:

**Information extraction:** NER can extract key information from textual data, allowing for better organization and retrieval of relevant details. This is particularly useful in the fields of journalism, legal document analysis, and academic research.

**Sentiment analysis:** By identifying named entities in text, NER can help determine the sentiment associated with specific entities, which can be valuable for understanding public opinion, brand reputation, and customer feedback.

**Knowledge graph construction:** NER can be used to identify entities and relationships in text, enabling the construction of knowledge graphs, which are useful for semantic search, question-answering systems, and artificial intelligence applications.

**Chatbots and conversational AI:** NER helps chatbots and virtual assistants identify and understand named entities mentioned in user input, allowing for more accurate and context-aware responses.


## Spacy

spaCy is a free and open-source library in Python designed for advanced Natural Language Processing (NLP). When working with large amounts of text, it's essential to gain insights, such as the text's topic, contextual meaning of words, entities involved in actions, and mentions of companies or products. spaCy is developed for production use, enabling the creation of applications that can process and "comprehend" vast amounts of textual data.

It can be employed to develop systems for information extraction or natural language understanding, or to prepare text for deep learning applications.


As shown in the following image, pre-trained models can extract common entities such as NORP, ORG, and PERSON. However, if you need to extract custom entities like skills from a resume, you'll have to create and train your own custom NER model.


<img src='/images/output_7_0_N.png'>  




```python
import spacy
import pandas as pd
from spacy import displacy
from spacy.matcher import Matcher
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from PIL import Image
import requests
from io import BytesIO
```

'**en_core_web_sm**' is english multi-task **CNN** trained on **OntoNotes**. Assigns context-specific token vectors, POS tags, dependency parse and named entities.

This is one of the pre-trained model we could use if want to extract only pre-define entities like PERSON, ORG etc.


```python
model = spacy.load('en_core_web_sm') #Pre-trained model
```


```python
doc = model('Hi I am Parsa and I was born on September 1996. \
           I work at a tech company in Toronto. I just bought a book \
           for $24 from Amazon. I love JAVA')
```

```python
displacy.render(doc,style='ent')
```


<div class="entities" style="line-height: 2.5; direction: ltr">Hi I am 
<mark class="entity" style="background: #aa9cfc; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Parsa
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PERSON</span>
</mark>
 and I was born on 
<mark class="entity" style="background: #bfe1d9; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    September 1996
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">DATE</span>
</mark>
.            I work at a tech company in 
<mark class="entity" style="background: #feca74; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Toronto
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">GPE</span>
</mark>
. I just bought a book for $
<mark class="entity" style="background: #e4e7d2; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    24
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">MONEY</span>
</mark>
 from 
<mark class="entity" style="background: #7aecec; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Amazon.
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">ORG</span>
</mark>

I love JAVA.</div>

---------------------

As you can see, the pre-trained models in spaCy can predict common entity types such as PERSON, DATE, and ORG, but they may not recognize custom entities like "JAVA" out of the box. To predict custom entities, you'll need to train your own NER model using annotated data that includes the custom entity types you're interested in.

## Spacy Pipeline

In spaCy, text processing starts by tokenizing the input, creating a Doc object. The Doc is then processed through several steps in a pipeline. The default pipeline used by spaCy's pretrained models includes a tagger, a parser, and an entity recognizer. Each pipeline component processes the Doc and returns it to be passed on to the next component.

The four main steps in the spaCy NLP pipeline are:

1. Tokenizer: This fixed component divides the text into individual tokens, such as words, punctuation, and whitespace. It is the first step in the pipeline and cannot be removed.
Tagger: The tagger assigns part-of-speech (POS) tags to each token based on the context and the token's position in the sentence.
2. Parser: The parser analyzes the grammatical structure of the text, providing information about dependencies and relationships between tokens, such as subject, object, and verb.
3. NER (Named Entity Recognition): The NER component identifies and classifies named entities in the text, such as people, organizations, locations, dates, and more.
While the tokenizer is a fixed step in the pipeline, the other three components (tagger, parser, and NER) can be removed or added according to the specific problem you want to solve. Custom components can also be created and integrated into the pipeline to extend its functionality.
    
<img src='/images/output_16_0_N.png'>  



```python
model.pipeline
```


    [('tagger', <spacy.pipeline.pipes.Tagger at 0x7fe888204e10>),
     ('parser', <spacy.pipeline.pipes.DependencyParser at 0x7fe88fac9590>),
     ('ner', <spacy.pipeline.pipes.EntityRecognizer at 0x7fe88fac9600>)]


Now, let's move on to creating a custom NER model using spaCy. To do this, we need to prepare our training data in the format required by spaCy. Unlike other machine learning algorithms, spaCy expects the training data to be formatted differently.

Our initial task is to convert our data into the required format for training. In the next cell, we'll directly use data in the appropriate format. However, in real-world applications, you'll need to annotate your data. To do this, you can use any annotation tool available in the market. Some popular annotation tools include:

Prodigy: A user-friendly, efficient annotation tool developed by the creators of spaCy. It offers an active learning system that helps users create training data more efficiently.
Doccano: An open-source text annotation tool that supports various types of annotations, including NER, sentiment analysis, and translation.



```python
TRAIN_DATA = [
   ("Python is great", {"entities": [(0, 6, "PROGLANG")]}),
   ("I like golang", {"entities": [(8, 14, "PROGLANG")]}),
   (("John codes in Java", {"entities": [(8, 14, "PROGLANG")]})),
   ('I used c++ project in production',{'entities': [(18, 21, 'PROGLANG')]}),
   ('Our company uses a Java server for our proxy',{'entities': [(17, 21, 'PROGLANG')]}) 
]
```


```python
TRAIN_DATA
```

    [('Python is cool', {'entities': [(0, 6, 'PROGLANG')]}),
     ('Me like golang', {'entities': [(8, 14, 'PROGLANG')]}),
     ('Yu like Java', {'entities': [(8, 14, 'PROGLANG')]}),
     ('How to set up unit testing for Visual Studio C++',
      {'entities': [(45, 48, 'PROGLANG')]}),
     ('How do you pack a visual studio c++ project for release?',
      {'entities': [(32, 35, 'PROGLANG')]}),
     ('How do you get leading wildcard full-text searches to work in SQL Server?',
      {'entities': [(62, 65, 'PROGLANG')]})]



## Creating Pipleline

To focus on the NER component and create a custom NER model for your specific problem, you can modify the spaCy pipeline accordingly.


```python
def create_blank_nlp(train_data):
    nlp = spacy.blank("en")
    ner = nlp.create_pipe("ner")
    nlp.add_pipe(ner, last=True)
    ner = nlp.get_pipe("ner")
    for _, annotations in train_data:
        for ent in annotations.get("entities"):
            ner.add_label(ent[2])
    return nlp  
```

## Training

Now that we have set up the pipeline with only the tokenizer and NER components, we can proceed with training our custom NER model. The performance of the model depends on the quality and quantity of the training data. This example demonstrates the training process, but the dataset used is small, so the final model performance may not be optimal.

```python
import random 
import datetime as dt

nlp = create_blank_nlp(TRAIN_DATA)
optimizer = nlp.begin_training()  
for i in range(10):
    random.shuffle(TRAIN_DATA)
    losses = {}
    for text, annotations in TRAIN_DATA:
        nlp.update([text], [annotations], sgd=optimizer, losses=losses)
    print(f"Losses at iteration {i} - {dt.datetime.now()}", losses)
```

    Losses at iteration 0 - 2020-05-11 14:13:25.916824 {'ner': 29.07171580195427}
    Losses at iteration 1 - 2020-05-11 14:13:26.074210 {'ner': 10.19973642288096}
    Losses at iteration 2 - 2020-05-11 14:13:26.231696 {'ner': 7.594819722430543}
    Losses at iteration 3 - 2020-05-11 14:13:26.395092 {'ner': 3.8065412295618444}
    Losses at iteration 4 - 2020-05-11 14:13:26.551974 {'ner': 0.6928493493463748}
    Losses at iteration 5 - 2020-05-11 14:13:26.702519 {'ner': 0.009406879480264863}
    Losses at iteration 6 - 2020-05-11 14:13:26.855381 {'ner': 2.0171781952037227e-05}
    Losses at iteration 7 - 2020-05-11 14:13:27.006226 {'ner': 2.1834541404970123e-06}
    Losses at iteration 8 - 2020-05-11 14:13:27.158256 {'ner': 5.645354740125843e-07}
    Losses at iteration 9 - 2020-05-11 14:13:27.311335 {'ner': 9.59412580172847e-08}


## Testing

Now that we've trained our custom NER model with the given training data, we can test its performance on unseen data. Since our training data does not include data points with "datascience" and "javascript," the model might not perform as well on these entities. 


```python
doc = nlp("I love to learn GOLANG") 
displacy.render(doc, style="ent")
```


<div class="entities" style="line-height: 2.5; direction: ltr">I love to learn 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    GOLANG
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PROGLANG</span>
</mark>
</div>



```python
doc = nlp("I never use django")
displacy.render(doc, style="ent")
```


<div class="entities" style="line-height: 2.5; direction: ltr">I never use 
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    django
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PROGLANG</span>
</mark>
</div>



```python
doc = nlp("Python is the best language.")
displacy.render(doc, style="ent")
```


<div class="entities" style="line-height: 2.5; direction: ltr">
<mark class="entity" style="background: #ddd; padding: 0.45em 0.6em; margin: 0 0.25em; line-height: 1; border-radius: 0.35em;">
    Python
    <span style="font-size: 0.8em; font-weight: bold; line-height: 1; border-radius: 0.35em; text-transform: uppercase; vertical-align: middle; margin-left: 0.5rem">PROGLANG</span>
</mark>
 is the best language</div>


