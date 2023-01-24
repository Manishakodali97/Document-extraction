def evaluate(ner_model, examples):
    scorer = Scorer()
    for input_, annot in examples:
        text = nlp.make_doc(input_)
        gold = Example.from_dict(text, annot)
        pred_value = ner_model(input_)
        scorer.score(gold)
    return scorer.scores

examples = " Give sample text data here to test the model performance "
ner_model = spacy.load('load your trained model') 
results = evaluate(ner_model, examples)
print(results)
