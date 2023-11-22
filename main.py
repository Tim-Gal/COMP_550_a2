from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize, pos_tag
from nltk.corpus import stopwords
import random
from loader import *


def preprocess_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    # Tokenize and lemmatize
    return [lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word.isalpha() and word not in stop_words]


def most_frequent_sense(dev_instances):
    baseline_predictions = {}
    for instance_id, instance in dev_instances.items():
        lemma = instance.lemma
        synsets = wn.synsets(lemma)
        baseline_predictions[instance_id] = synsets[0].lemmas()[0].key() if synsets else None

    return baseline_predictions


def calculate_accuracy(predictions, gold_standard):
    correct_predictions = sum(pred in gold_standard[i] for i, pred in predictions.items())
    accuracy = correct_predictions / len(gold_standard)
    return accuracy


def lesk_algorithm(dev_instances):
    lesk_predictions = {}
    for instance_id, instance in dev_instances.items():
        context = preprocess_sentence(' '.join(instance.context))
        lesk_sense = lesk(context, instance.lemma)
        if lesk_sense:
            # Find the lemma in the synset that matches the target word
            matched_lemma = next((lemma for lemma in lesk_sense.lemmas() if lemma.name() == instance.lemma), None)
            if matched_lemma:
                # Get the sense key for the matched lemma
                lesk_predictions[instance_id] = matched_lemma.key()

    return lesk_predictions


def bootstrap_lesk(dev_instances, iterations=5):
    # Starting with the most frequent sense as the seed
    seed_instances = most_frequent_sense(dev_instances)

    # Bootstrapping iterations
    for iteration in range(iterations):
        # Apply Lesk's algorithm and collect predictions
        bootstrap_lesk_predictions = {}
        for instance_id, instance in dev_instances.items():
            context = preprocess_sentence(' '.join(instance.context))
            # If we already have a prediction from the seed, use that
            if instance_id in seed_instances:
                lesk_sense = wn.lemma_from_key(seed_instances[instance_id]).synset()
            else:
                # Otherwise, use Lesk's algorithm to predict
                lesk_sense = lesk(context, instance.lemma)

            if lesk_sense:
                matched_lemma = next((lemma for lemma in lesk_sense.lemmas() if lemma.name() == instance.lemma), None)
                if matched_lemma:
                    bootstrap_lesk_predictions[instance_id] = matched_lemma.key()

    return bootstrap_lesk_predictions


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}

    # Implement the most frequent sense baseline and collect predictions
    baseline_predictions = most_frequent_sense(dev_instances)

    # Calculate baseline accuracy
    baseline_accuracy = calculate_accuracy(baseline_predictions, dev_key)
    print(f"Baseline Accuracy: {baseline_accuracy:.2%}")

    # Apply Lesk's algorithm and collect predictions
    lesk_predictions = lesk_algorithm(dev_instances)

    # Calculate Lesk accuracy
    lesk_accuracy = calculate_accuracy(lesk_predictions, dev_key)
    print(f"Lesk Accuracy: {lesk_accuracy:.2%}")

    # Apply Lesk's algorithm with bootstrapping and
    bootstrap_lesk_predictions = bootstrap_lesk(dev_instances, dev_key)

    # Calculate bootstrap Lesk accuracy
    bootstrap_lesk_accuracy = calculate_accuracy(bootstrap_lesk_predictions, dev_key)
    print(f"Lesk Accuracy: {bootstrap_lesk_accuracy:.2%}")
