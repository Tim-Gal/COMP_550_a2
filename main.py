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


def most_frequent_sense(lemma):
    synsets = wn.synsets(lemma)
    return synsets[0].lemmas()[0].key() if synsets else None


def apply_lesk_algorithm(context_sentence, ambiguous_word):
    return lesk(context_sentence, ambiguous_word)


def calculate_accuracy(predictions, gold_standard):
    correct_predictions = sum(pred in gold_standard[i] for i, pred in predictions.items())
    accuracy = correct_predictions / len(gold_standard)
    return accuracy


def bootstrap_lesk(dev_instances, dev_key, iterations=5):
    # Starting with the most frequent sense as the seed
    seed_instances = {}
    for instance_id, instance in dev_instances.items():
        lemma = instance.lemma
        seed_instances[instance_id] = most_frequent_sense(lemma)

    # Bootstrapping iterations
    for iteration in range(iterations):
        # Apply Lesk's algorithm and collect predictions
        lesk_predictions = {}
        for instance_id, instance in dev_instances.items():
            context = preprocess_sentence(' '.join(instance.context))
            # If we already have a prediction from the seed, use that
            if instance_id in seed_instances:
                lesk_sense = wn.lemma_from_key(seed_instances[instance_id]).synset()
            else:
                # Otherwise, use Lesk's algorithm to predict
                lesk_sense = apply_lesk_algorithm(context, instance.lemma)

            if lesk_sense:
                matched_lemma = next((lemma for lemma in lesk_sense.lemmas() if lemma.name() == instance.lemma), None)
                if matched_lemma:
                    lesk_predictions[instance_id] = matched_lemma.key()

    # Evaluate predictions
    accuracy = calculate_accuracy(lesk_predictions, dev_key)
    print(f"Iteration {iteration}: Lesk Accuracy: {accuracy:.2%}")

    # Select a subset of predictions to add to the seed set
    confident_predictions = {k: v for k, v in lesk_predictions.items() if
                             random.random() > 0.5}  # random confidence simulation

    # Update the seed set with new confident predictions
    seed_instances.update(confident_predictions)


if __name__ == '__main__':
    data_f = 'multilingual-all-words.en.xml'
    key_f = 'wordnet.en.key'
    dev_instances, test_instances = load_instances(data_f)
    dev_key, test_key = load_key(key_f)

    # IMPORTANT: keys contain fewer entries than the instances; need to remove them
    dev_instances = {k: v for (k, v) in dev_instances.items() if k in dev_key}
    test_instances = {k: v for (k, v) in test_instances.items() if k in test_key}

    # read to use here
    print(len(dev_instances))  # number of dev instances
    print(len(test_instances))  # number of test instances

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

    # Apply Lesk's algorithm with bootstrapping and
    bootstrap_lesk(dev_instances, dev_key)
