# keras-nlp-snippets
Short snippets of different nlp AI tasks written with Keras.


The scripts gradually get more complex. Links regarding basic ideas, code examples or concepts are left in the comments in each script.
Use _load_data.py to load yml files (here starting with "conversations").

The data directory is not included. Here are the links
- shubham0204/chatbot_nlp: https://github.com/shubham0204/Dataset_Archives/blob/master/chatbot_nlp.zip
- glove.6B.100d.txt: https://www.kaggle.com/terenceliu4444/glove6b100dtxt (https://nlp.stanford.edu/projects/glove/)
- german_vectors_100.txt: https://deepset.ai/german-word-embeddings (https://devmount.github.io/GermanWordEmbeddings/)


To use embedding n char level (i.e. german_vectors_100_char.txt as in 2_2_text_generation_simple_char.py), use utils/glove_char_embeddings.py on your glove embedding file.


### 0 Text classification
Just simple classifiers.

### 1 Sequence 2 sequence
Used i.e. for Q&A or translation, basically everything following the schema: something in -> something else out

### 2 Text generation
Basic text generation based on recipes separated by linebreaks:
1) on char level: 2_2_text_generation_simple_char.py
2) on word level: 2_3_text_generation_simple_word.py
