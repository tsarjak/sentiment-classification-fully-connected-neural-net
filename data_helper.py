import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import pickle
import numpy as np
import random
from collections import Counter

lemmatizer = WordNetLemmatizer()
max_lines = 100000

def lexiconInit(pos,neg) :
	lex = []

	# Tokenizing the words and storing it in a list
	with open(pos,'r') as f:
		lines = f.readlines()
		for l in lines[:max_lines]:
			words = word_tokenize(l.lower())
			lex += list(words)

	with open(neg,'r') as f:
		lines = f.readlines()
		for l in lines[:max_lines]:
			words = word_tokenize(l.lower())
			lex += list(words)

	# Lemmatizing the words in the lexicon list
	lex = [lemmatizer.lemmatize(i) for i in lex]

	#Counting and removind unneccesary words (Optional. Try without this to see the effect on training)
	wordCounts = Counter(lex)
	lexWithCount = []
	for w in wordCounts :
		if 1000 > wordCounts[w] > 40 :
			lexWithCount.append(w)

	print("Lexicon Initialization Complete")
	return lexWithCount


def sampler(fileLoc, lexicon, classification):

	featSet = []

	with open(fileLoc,'r') as f:
		lines = f.readlines()
		for l in lines:
			currWords = word_tokenize(l.lower())
			currWords = [lemmatizer.lemmatize(i) for i in currWords]
			feat = np.zeros(len(lexicon))
			for word in currWords:
				if word.lower() in lexicon:
					feat[lexicon.index(word.lower())] += 1

			feat = list(feat)
			featSet.append(feat)

	return featSet


def createFeatureSets(pos,neg,testProp = 0.1):

	lexicon = lexiconInit(pos,neg)
	feats = []
	feats += sampler('pos.txt',lexicon,[1,0])
	feats += sampler('neg.txt',lexicon,[0,1])
	random.shuffle(feats)
	feats = np.array(feats)


	testProp = int(testProp*len(feats))
	train_x = list(feats[:,0][:-testProp])
	train_y = list(feats[:,1][:-testProp])
	test_x = list(feats[:,0][-testProp:])
	test_y = list(feats[:,1][-testProp:])

	return train_x,train_y,test_x,test_y


def main():
	train_x,train_y,test_x,test_y = createFeatureSets('pos.txt','neg.txt')
	with open('processedData.pickle','wb') as f:
		pickle.dump([train_x,train_y,test_x,test_y],f)

if __name__ == '__main__':
    main()