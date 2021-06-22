from contextualized_topic_models.models.ctm import ZeroShotTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
import nltk
import numpy as np
from contextualized_topic_models.evaluation.measures import *

nltk.download('stopwords')

text_file = "dbpedia_sample_abstract_20k_unprep.txt"
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
print(ctm.get_topic_lists(5))
# measurement on english,
# input: topics, documents
cNPMI = CoherenceNPMI(
    ctm.get_topic_lists(10), [d.split() for d in preprocessed_documents])
print("cNPMI score for English: ", cNPMI.score())

# process italian
italian_documents = [
    line.strip() for line in open("italian_documents.txt", 'r').readlines()
]
testing_dataset = tp.transform(
    italian_documents)  # create dataset for the testset
# n_sample how many times to sample the distribution (see the documentation)
italian_topics_predictions = ctm.get_thetas(
    testing_dataset, n_samples=5)  # get all the topic predictions
# get the topic id of the first document
topic_number = np.argmax(italian_topics_predictions[0])
print(ctm.get_topic_lists(10)[topic_number])

# docuemnts should be the English translation of the italian documents
cNPMI = CoherenceNPMI(
    ctm.get_topic_lists(10), [d.split() for d in italian_documents])
print("cNPMI score for Italian: ", cNPMI.score())
