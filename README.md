# Document-extraction
SpaCy provides an exceptionally efficient statistical system for NER in python, which can assign labels to groups of tokens which are contiguous. It provides a default model which can recognize a wide range of named or numerical entities, which include person, organization, language, event etc. Apart from these default entities, spaCy also gives us the liberty to add arbitrary classes to the NER model, by training the model to update it with newer trained examples.

Load the model 1.1. spacy.load('en') --> Disable existing pipe line (nlp.disable_pipes) 1.2. spacy.blank('en') --> Added Entity Recognizer to Pipeline
Shuffle and loop over the examples --> update the model (nlp.update)
Save the trained model (nlp.to_disk)
Test the model performance by test.py file
