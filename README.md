# Code for analyzing hate speech using knowledge embeddings and BERT representation

This is code I wrote during my internship at Spoken Language Systems, Saarland University. I worked on hate speech detection, and the dataset used was the OffensEval hate speech corpus released in SemEval 2019. I took some references and code from these repos/websites:

[OffensEval Code](https://github.com/ZeyadZanaty/offenseval/)
[Doc2Vec training](https://markroxor.github.io/gensim/static/notebooks/doc2vec-wikipedia.html)
[BERTweet](https://github.com/VinAIResearch/BERTweet)

The steps involved are roughly this, if you wish to reproduce the code:
1. Place dataset (in csv/tsv format) in the /Data folder.
2. Run ```GenerateBERT.py``` to generate BERT embeddings and save them in /pickles folder.
3. Run either of the three files in /Entity Extraction to generate entities/noun phrases from tweets. (The code in ```Stanford.py``` needs to be added to more.) 
4. Run ```TrainDoc2Vec.py``` to train two Doc2Vec models based on Wiki corpus for Wikipedia embeddings. (Contact me for pretrained model files.)
5. Run ```EntityEmbeddings.py``` to generate embeddings from extracted entities using trained doc2vec models.
6. Finally, run either of ```TrainSVM.py``` and ```TrainRNN.py``` to train different models and see the outcome.

Some tips:
1. Adjust the paths for saving and loading files everywhere.
2. In the code I took from the links above, I have made significant edits. Feel free to remove them/add more to them to further investigate the system.
