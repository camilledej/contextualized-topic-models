from contextualized_topic_models.models.ctm import CombinedTM
from contextualized_topic_models.utils.data_preparation import TopicModelDataPreparation
from contextualized_topic_models.utils.preprocessing import WhiteSpacePreprocessing
from contextualized_topic_models.evaluations.measures import *
import nltk

text_file = "dbpedia_sample_abstract_20k_unprep.txt"
nltk.download('stopwords')

documents = [
    line.strip() for line in open(text_file, encoding="utf-8").readlines()
]
sp = WhiteSpacePreprocessing(documents, stopwords_language='english')
preprocessed_documents, unpreprocessed_corpus, vocab = sp.preprocess()

tp = TopicModelDataPreparation("paraphrase-distilroberta-base-v1")

training_dataset = tp.fit(
    text_for_contextual=unpreprocessed_corpus,
    text_for_bow=preprocessed_documents)

print(tp.vocab[:10])

# topics, docs
#cNPMI = CoherenceNPMI()
