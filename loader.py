"""
@author: jcheung

Developed for Python 2. Automatically converted to Python 3; may result in bugs.
"""
import xml.etree.cElementTree as ET
import codecs
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords


class WSDInstance:
    def __init__(self, my_id, lemma, context, index):
        self.id = my_id  # id of the WSD instance
        self.lemma = lemma  # lemma of the word whose sense is to be resolved
        self.context = context  # lemma of all the words in the sentential context
        self.index = index  # index of lemma within the context

    def __str__(self):
        """
        For printing purposes.
        """
        return '%s\t%s\t%s\t%d' % (self.id, self.lemma, ' '.join(self.context), self.index)


def load_instances(f):
    """
    Load two lists of cases to perform WSD on. The structure that is returned is a dict, where
    the keys are the ids, and the values are instances of WSDInstance.
    """
    tree = ET.parse(f)
    root = tree.getroot()

    dev_instances = {}
    test_instances = {}

    for text in root:
        if text.attrib['id'].startswith('d001'):
            instances = dev_instances
        else:
            instances = test_instances
        for sentence in text:
            # construct sentence context
            context = [to_ascii(el.attrib['lemma']) for el in sentence]
            for i, el in enumerate(sentence):
                if el.tag == 'instance':
                    my_id = el.attrib['id']
                    lemma = to_ascii(el.attrib['lemma'])
                    instances[my_id] = WSDInstance(my_id, lemma, context, i)
    return dev_instances, test_instances


def load_key(f):
    """
    Load the solutions as dicts.
    Key is the id
    Value is the list of correct sense keys.
    """
    dev_key = {}
    test_key = {}
    for line in open(f):
        if len(line) <= 1: continue
        # print (line)
        doc, my_id, sense_key = line.strip().split(' ', 2)
        if doc == 'd001':
            dev_key[my_id] = sense_key.split()
        else:
            test_key[my_id] = sense_key.split()
    return dev_key, test_key


def to_ascii(s):
    # remove all non-ascii characters
    return codecs.encode(s, 'ascii', 'ignore').decode('ascii')


def preprocess_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Tokenize and lemmatize
    return [lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word.isalpha() and word not in stop_words]


def most_frequent_sense(lemma):
    synsets = wn.synsets(lemma)
    return synsets[0].lemmas()[0].key() if synsets else None


def apply_lesk_algorithm(context_sentence, ambiguous_word):
    return lesk(context_sentence, ambiguous_word)


def calculate_accuracy(predictions, gold_standard):
    correct_predictions = sum(pred in gold_standard[i] for i, pred in predictions.items())
    accuracy = correct_predictions / len(gold_standard)
    return accuracy


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}

    # read to use here
    #print(len(dev_instances))  # number of dev instances
    #print(len(test_instances))  # number of test instances

    # Implement the most frequent sense baseline and collect predictions
    baseline_predictions = {}
    for instance_id, instance in dev_instances.items():
        lemma = instance.lemma
        baseline_predictions[instance_id] = most_frequent_sense(lemma)
    print(baseline_predictions)

    # Calculate baseline accuracy
    baseline_accuracy = calculate_accuracy(baseline_predictions, dev_key)
    print(f"Baseline Accuracy: {baseline_accuracy:.2%}")

    # Apply Lesk's algorithm and collect predictions
    lesk_predictions = {}
    for instance_id, instance in dev_instances.items():
        context = preprocess_sentence(' '.join(instance.context))
        lesk_sense = apply_lesk_algorithm(context, instance.lemma)
        if lesk_sense:
            # Find the lemma in the synset that matches the target word
            matched_lemma = next((lemma for lemma in lesk_sense.lemmas() if lemma.name() == instance.lemma), None)
            if matched_lemma:
                # Get the sense key for the matched lemma
                lesk_predictions[instance_id] = matched_lemma.key()
    print(lesk_predictions)

    # Calculate Lesk accuracy
    lesk_accuracy = calculate_accuracy(lesk_predictions, dev_key)
    print(f"Lesk Accuracy: {lesk_accuracy:.2%}")
