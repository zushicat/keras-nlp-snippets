# keras-nlp-snippets
Short snippets of different nlp AI tasks written with Keras.


The scripts gradually get more complex. Links regarding basic ideas, code examples or concepts are left in the comments in each script.
Use _load_data.py to load yml files (here starting with "conversations").

The data directory is not included. Here are the links
- shubham0204/chatbot_nlp: https://github.com/shubham0204/Dataset_Archives/blob/master/chatbot_nlp.zip
- glove.6B.100d.txt: https://www.kaggle.com/terenceliu4444/glove6b100dtxt (https://nlp.stanford.edu/projects/glove/)
- german_vectors_100.txt: https://deepset.ai/german-word-embeddings (https://devmount.github.io/GermanWordEmbeddings/)


Due to copyright reasons there are no links available for crawled recipe texts and titles.


### 0 Text classification
Classify text.

### 1 Sequence 2 sequence
Used i.e. for Q&A or translation, basically everything following the schema: something in -> something else out

### 2 Text generation
Text generation
1) on char level: 2_2_text_generation_simple_char.py
2) on word level: 2_3_text_generation_simple_word.py


#### 2.1.1 Generating german dessert recipes with script 1)
- shortened input file with 91 dessert recipe texts (from hundreds)
- pretrained char weights (glove)
- 3 lstm layers with 128 hidden units
- 700 epochs (loss ≈ 0.4)

Results could be worse but are certainly not satisfying. 

Examples:
```
seed: "vorsichtig in eine schüssel geben"

vorsichtig in eine schüssel geben. auf dem salz steif schlagen, dann alles in kleine schälchen geben. mit zimtzucker bestreuen. die masse in portionsweise vierteln, das fruchtfleisch heraus belieben mit vanillezucker glatt rühren und die mascarponecreme unterrühren und mit einem kräcken mit dem vanillezucker verrühren und kalt stellen. die mandeln in einer pfanne rett mit dem zucker und die frucht darunter ziehen. eine teller glatt streichen. dann die früchte zusammen mit den saft mit dem saft der früchte zu einer schüssel geben und mit den johandisbee zund. stamit leicht ziehen lassen. die apfelkücherl ein gestellt wasser und den zucker und die pfanne erhitzen, den teig nehmen. zum schluss das fruchtfleisch mit einem schneebesen und die kokosmilch und dem saft vermischen, unterheben.
```

```
seed: "den zucker einrieseln lassen und"

den zucker einrieseln lassen und 2 min. köcheln lassen. die milch und den zucker und scheiben schneiden. die mascarpone mit dem saft der früchte darin bei schwicherpaben, servieren langsam die milch unter den teig ziehen. für das dressing darüber geben und mit einem schälen und dann aus der kokosrüssel servieren.
```

Suggestions for another attempt:
- more text input
- more hidden units for the lstm layers
- train more epochs


#### 2.2 Generating german recipes with script 2)
- use all dessert texts
- pretrained word weights (glove)
- (for starters) only 1 lstm layer with 128 hidden units

