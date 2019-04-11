#!/usr/bin/env python
# coding: utf-8

import re
import nltk
import contractions
import pandas as pd

from nltk.corpus import wordnet
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import sent_tokenize, word_tokenize
from sacremoses import MosesTokenizer, MosesDetokenizer

from sklearn.pipeline import Pipeline


################ Importation des datas ###################""


df = pd.read_csv('Data/dataset.csv', sep='\t', header = None, names = ["Avis"],encoding = "ISO-8859-1")
labels = pd.read_csv('Data/labels.csv', sep='\t', header = None, names = ['Note'],encoding = "ISO-8859-1")
all = pd.concat([df.reset_index(drop=True),labels.reset_index(drop=True)], axis=1)
sample=all.sample(n=100).reset_index()
df=sample.Avis
df=df.to_frame()
labels=sample.Note
labels=labels.to_frame()


dfc = df.Avis.copy()


############### Premiers traitements #################

def regex(dfc) : 

	for i in range(0, len(dfc), 1): 

		dfc[i] = re.sub(r"\d", "", dfc[i]) # remove chiffre
		dfc[i] = re.sub(r"\s+[a-zA-Z]\s+", " ", dfc[i]) # remove single char
		dfc[i] = re.sub(r'\S+@\S+', " ", dfc[i]) # remove email
		dfc[i] = re.sub(r"\s+"," ", dfc[i], flags = re.I)   # remove extra spaces
		dfc[i] = re.sub(r"\."," ",dfc[i]) # remove les points
	
	return dfc

######## Tokenisation et premiers traitements #################

lemmatizer = WordNetLemmatizer()

def tokentag(dfc) : 

	for i in range(0, len(dfc), 1): 
		
		dfc[i]=''.join(dfc[i]).lower() #jusqu'ici c'est une liste
		dfc[i]=sent_tokenize(dfc[i])# ici cast de la liste en string(avis entier)
    	
		for j in range(0, len(dfc.loc[i]), 1):
        
			dfc.loc[i][j] = contractions.fix(dfc.loc[i][j]) #retrait contractions
			dfc.loc[i][j] = dfc.loc[i][j].lower() # mise en minuscule
			dfc.loc[i][j] = word_tokenize(dfc.loc[i][j]) # tokenisation
        	
			sentence = dfc.loc[i][j]
        
			for word in sentence:

				if (word == ".") :
					word.replace(word," ") # Suppressions des .

			dfc.loc[i][j] = nltk.pos_tag(sentence) # pose des tags


        
############# Lemma en fonction des tag ##################


def get_wordnet_pos(word,pos):
    #tag = nltk.pos_tag([word])[0][1][0].upper()
    tag_dict = {"JJ": wordnet.ADJ,
    			"JJR": wordnet.ADJ,
    			"JJS": wordnet.ADJ,
                "NN": wordnet.NOUN,
                "NNS": wordnet.NOUN,
                "NNP": wordnet.NOUN,
                "NNPS": wordnet.NOUN,
                "VB": wordnet.VERB,
                "VBD": wordnet.VERB,
                "VBG": wordnet.VERB,
                "VBN": wordnet.VERB,
                "VBP": wordnet.VERB,
                "VBZ": wordnet.VERB,
                "RB": wordnet.ADV,
                "RBS": wordnet.ADV,
                "RBR": wordnet.ADV}
    return tag_dict.get(pos, wordnet.NOUN)


def lema(dfc) :

	for i in range(0, len(dfc), 1): 

		for j in range(0, len(dfc.loc[i]), 1):

			k = 0
			sentence = dfc[i][j]
			dfc[i][j] = ([(lemmatizer.lemmatize(w, get_wordnet_pos(w,pos)),pos) for w,pos in sentence])

	return dfc




############### Suppression des mots ballec #################

NLTKSW = set(['i',':',',',';',')','(,','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves','he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves','what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being','have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while','of','at','by','for','with','about','against','between','into','through','during','before','after','above','below','to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where','why','how','all','any','both','each','few','more','most','other','some','such','only','own','same','so','than','too','very','s','t','can','will','just','don','should','now'])
NLTK_LIST = [',','.','$','``',"''",'(',')',':','CC','CD','DT','EX','FW','IN','LS','SYM','TO','WP',r'NPP.?','POS','PDT',r'PRP.?','RP','WTD','WP','WRB','"']


def trash(dfc) :

	for i in range(0, len(dfc), 1): 

		for j in range(0, len(dfc.loc[i]), 1):

			dfc[i][j] = ([[word,pos] for word,pos in dfc[i][j] if ((pos not in NLTK_LIST) and (word not in NLTKSW) and (word.isalpha()))])

################### Remise sous forme initiale ##################

def detokenize(dfc) :

	#from nltk.tokenize.moses import MosesDetokenizer

	detokenizer = MosesDetokenizer()

	for i in range(0, len(dfc), 1):
		for j in range(0, len(dfc.loc[i]), 1):
			dfc[i][j] = ([word for word,pos in dfc[i][j]])
			dfc[i][j] = detokenizer.detokenize(dfc[i][j], return_str=True)
		dfc[i] = detokenizer.detokenize(dfc[i], return_str=True)
    

def clean(dfc) :
	regex(dfc)
	tokentag(dfc)
	lema(dfc)
	trash(dfc)
	detokenize(dfc)

	return dfc


clean(dfc)
print(dfc[1])