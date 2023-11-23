from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.stem import WordNetLemmatizer
from nltk import word_tokenize
from nltk.corpus import stopwords
import random
from loader import *


def preprocess_sentence(sentence):
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    return [lemmatizer.lemmatize(word) for word in word_tokenize(sentence) if word.isalpha() and word not in stop_words]


def calculate_accuracy(predictions, gold_standard):
    correct_predictions = sum(pred in gold_standard[i] for i, pred in predictions.items())
    accuracy = correct_predictions / len(gold_standard)
    return accuracy


def most_frequent_sense(dev_instances):
    baseline_predictions = {}
    for instance_id, instance in dev_instances.items():
        lemma = instance.lemma
        synsets = wn.synsets(lemma)
        baseline_predictions[instance_id] = synsets[0].lemmas()[0].key() if synsets else None

    return baseline_predictions


def lesk_algorithm(dev_instances):
    lesk_predictions = {}
    for instance_id, instance in dev_instances.items():
        context = preprocess_sentence(' '.join(instance.context))
        lesk_sense = lesk(context, instance.lemma)
        if lesk_sense:
            # find the lemma in the synset that matches the target word
            matched_lemma = next((lemma for lemma in lesk_sense.lemmas() if lemma.name() == instance.lemma), None)
            if matched_lemma:
                # get the sense key for the matched lemma
                lesk_predictions[instance_id] = matched_lemma.key()

    return lesk_predictions


def bootstrap_lesk(dev_instances, iterations=5):
    # start with the most frequent sense
    seed_instances = most_frequent_sense(dev_instances)

    bootstrap_lesk_predictions = {}
    for iteration in range(iterations):
        bootstrap_lesk_predictions = {}
        for instance_id, instance in dev_instances.items():
            context = preprocess_sentence(' '.join(instance.context))
            # if there's already a prediction from the seed, then use it
            if instance_id in seed_instances:
                lesk_sense = wn.lemma_from_key(seed_instances[instance_id]).synset()
            else:
                # otherwise, use Lesk's algorithm to predict
                lesk_sense = lesk(context, instance.lemma)

            # find the lemma that matches the instance's lemma in Lesk sense
            matched_lemma = next((lemma for lemma in lesk_sense.lemmas() if lemma.name() == instance.lemma), None)
            bootstrap_lesk_predictions[instance_id] = matched_lemma.key() if matched_lemma else None

    return bootstrap_lesk_predictions


def apply_random_sense_selection(dev_instances):
    random_predictions = {}
    for instance_id, instance in dev_instances.items():
        senses = wn.synsets(instance.lemma)
        selected_sense = random.choice(senses) if senses else None
        # if a random sense was selected, add its key
        if selected_sense:
            random_predictions[instance_id] = selected_sense.lemmas()[0].key()
    return random_predictions


def enhanced_lesk_algorithm(dev_instances):
    lesk_predictions = {}
    for instance_id, instance in dev_instances.items():
        context = preprocess_sentence(' '.join(instance.context))

        best_sense = None
        max_overlap = 0

        for sense in wn.synsets(instance.lemma):
            # create a signature for the sense
            signature = set(preprocess_sentence(sense.definition()))
            for example in sense.examples():
                signature.update(preprocess_sentence(example))
            for synonym in sense.lemmas():
                signature.add(synonym.name().replace('_', ' '))
                signature.update([lemma.name().replace('_', ' ') for lemma in synonym.synset().lemmas() if lemma.name() != synonym.name()])

            # calculate the overlap between context and signature
            overlap = len(set(context).intersection(signature))

            # update sense if overlap is higher
            if overlap > max_overlap:
                max_overlap = overlap
                best_sense = sense

        lesk_sense = best_sense

        if lesk_sense:
            matched_lemma = next((lemma for lemma in lesk_sense.lemmas() if lemma.name() == instance.lemma), None)
            if matched_lemma:
                lesk_predictions[instance_id] = matched_lemma.key()

    return lesk_predictions


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}

    # apply the most frequent sense baseline
    baseline_predictions = most_frequent_sense(dev_instances)
    # calculate baseline accuracy
    baseline_accuracy = calculate_accuracy(baseline_predictions, dev_key)
    print(f"Baseline Accuracy: {baseline_accuracy:.2%}")

    # apply Lesk's algorithm
    lesk_predictions = lesk_algorithm(dev_instances)
    # calculate Lesk accuracy
    lesk_accuracy = calculate_accuracy(lesk_predictions, dev_key)
    print(f"Lesk Accuracy: {lesk_accuracy:.2%}")

    # apply Lesk's algorithm with bootstrapping and
    bootstrap_lesk_predictions = bootstrap_lesk(dev_instances)
    # calculate bootstrap Lesk accuracy
    bootstrap_lesk_accuracy = calculate_accuracy(bootstrap_lesk_predictions, dev_key)
    print(f"Bootstrap Lesk Accuracy: {bootstrap_lesk_accuracy:.2%}")

    # apply random selection
    random_predictions = apply_random_sense_selection(dev_instances)
    # calculate random selection accuracy
    random_accuracy = calculate_accuracy(random_predictions, dev_key)
    print(f"Random Accuracy: {random_accuracy:.2%}")

    # apply enhanced version of Lesk's algorithm
    enhanced_predictions = enhanced_lesk_algorithm(dev_instances)
    # calculate enhanced Lesk accuracy
    enhanced_accuracy = calculate_accuracy(enhanced_predictions, dev_key)
    print(f"Enahnced Lesk Accuracy: {enhanced_accuracy:.2%}")
