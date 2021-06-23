from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
import nltk
import numpy as np
from contextualized_topic_models.evaluation.measures import *
import pickle
import os
os.environ["TOKENIZERS_PARALLELISM"] = "true"

nltk.download('stopwords')

text_file = "dbpedia_train_unprep.txt"
documents = [
    line.strip() for line in open(text_file, encoding="utf-8").readlines()
]
sp = WhiteSpacePreprocessing(documents, stopwords_language='english')
preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()

tp = TopicModelDataPreparation("distiluse-base-multilingual-cased")

training_dataset = tp.fit(
    text_for_contextual=unpreprocessed_corpus,
    text_for_bow=preprocessed_documents)

print(tp.vocab[:10])

# train zero shot model
ctm = ZeroShotTM(
    bow_size=len(tp.vocab),
    contextual_size=512,
    n_components=50,
    num_epochs=20)
ctm.fit(training_dataset)  # run the model
print("ctm.get_topic_lists: ", ctm.get_topic_lists(5))

#save model
#filename = 'finalized_model.sav'
#pickle.dump(ctm, open(filename,'wb')

#load model
#loaded_model = pickle.load(open(filename,'rb'))
#print("loaded_model.get_topic_lists: ",loaded_model.get_topic_lists(5))

# measurement on english,
# input: topics, documents
with open(r"test_set","rb") as f:
	data = pickle.load(f)
print("data[0][0]: ",data[0][0])
print("len data: ", len(data))
print("type data: ", type(data))
cNPMI = CoherenceNPMI(
    ctm.get_topic_lists(10), [d.split() for d in preprocessed_documents])
print("trained cNPMI score for English: ", cNPMI.score())
#cNPMI = CoherenceNPMI(
#    loaded_model.get_topic_lists(10), [d.split() for d in preprocessed_documents])
print("loaded cNPMI score for English: ",cNPMI.score())

# process italian
italian_documents = [item[0] for item in data]
print("len italian_documents: ", len(italian_documents))
print("type italian_documents: ", type(italian_documents))
print("italian_documents[0]: ",italian_documents[0])

italian_testing_dataset = tp.transform(
    italian_documents)  # create dataset for the testset
# n_sample how many times to sample the distribution (see the documentation)
italian_topics_predictions = ctm.get_thetas(
    italian_testing_dataset, n_samples=5)  # get all the topic predictions
# get the topic id of the first document
topic_number = np.argmax(italian_topics_predictions[0])
print("Italian test sample: ", ctm.get_topic_lists(10)[topic_number])

#test english
english_documents = [item[4] for item in data]
print("len english_documents: ",len(english_documents))
print("type english_documents: ",type(english_documents))
print("english_documents[0]: ", english_documents[0])
english_testing_dataset = tp.transform(
	english_documents)
english_topics_predictions = ctm.get_thetas(
	english_testing_dataset, n_samples=5) #get all the topic predictions
topic_number = np.argmax(english_topics_predictions[0])
print("English test sample: ", ctm.get_topic_lists(10)[topic_number])

#Get document distributions, for use in all metrics
doc_dist_eng = ctm.get_doc_topic_distribution(english_testing_dataset,n_samples=100)
doc_dist_ital = ctm.get_doc_topic_distribution(italian_testing_dataset,n_samples=100) 

#Check Eng-Ital matches in test data
matchItal = Matches(doc_dist_eng,doc_dist_ital)
print("English-Italian match score: ", matchItal.score())

## Eng-Ital divergence
divItal = KLDivergence(doc_dist_eng,doc_dist_ital)
print("English-Italian divergence score: ",divItal.score())

##Eng-Ital centroid distance
centdistItal = CentroidDistance(doc_dist_eng,doc_dist_ital,
	topics=ctm.get_topic_lists(10),
	word2vec_path=None,binary=True,topk=10)
print("English-Italian centdist score: ",centdistItal.score())
