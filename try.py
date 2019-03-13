from nltk import ngrams

sentence = 'this is a foo bar sentences and i want to ngramize it'

n = 6
sixgrams = ngrams(sentence.split(), n)
ansgrams=[]
for grams in sixgrams:
    ansgrams.append(' '.join(grams))
print(ansgrams)