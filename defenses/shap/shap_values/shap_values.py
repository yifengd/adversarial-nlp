import logging
import math
import numpy as np
import pandas as pd
import shap
import torch
from sklearn.utils import shuffle
from transformers import AutoTokenizer
from transformers import AutoModelForSequenceClassification

logger = logging.getLogger("attackdetect")
logger.setLevel(logging.INFO)


def predict_labels(
    data, model_name="textattack/bert-base-uncased-SST-2", bert_type="bert-base-uncased"
):
    model = AutoModelForSequenceClassification.from_pretrained(model_name)
    model.cuda() if torch.cuda.is_available() else model.cpu()
    tokenizer = AutoTokenizer.from_pretrained(bert_type)

    # Tokenize inputs
    embeddings = tokenizer(list(data), return_tensors="pt", padding=True).input_ids
    # Predict
    prediction = (
        model(embeddings.cuda() if torch.cuda.is_available() else embeddings.cpu())
        .logits.detach()
        .cpu()
        .numpy()
    )
    # Normalise to derive scores
    scores = (np.exp(prediction).T / np.exp(prediction).sum(-1)).T
    # Use sigmoid to select max score
    val = [1 / (1 + math.exp(-x)) for x in scores[:, 1]]
    return val


def transform_attack_data(adv_examples):
    # Filter out skipped and unsuccessful attacks
    adv_examples = adv_examples[adv_examples["result_type"] == "Successful"]

    if len(adv_examples) == 0:
        raise ValueError("No successful attacks found")

    orig_x_test = np.array(adv_examples["original_text"])
    orig_y_test = np.array(adv_examples["original_output"])
    adv_x_test = np.array(adv_examples["perturbed_text"])
    adv_y_test = np.array(adv_examples["perturbed_output"])

    return orig_x_test, orig_y_test, adv_x_test, adv_y_test


def pad_embeddings(shap_values, max_tokens: int = 200):
    padded = []

    for sentence in shap_values.values:
        padded.append(np.pad(sentence[:max_tokens], (0, max_tokens - sentence.size)))

    return np.array(padded)


def main(params: dict):
    tokenizer = AutoTokenizer.from_pretrained(params.bert_type)
    attack_data = pd.read_table(params.attack_file_path).head(int(params.num_sentences))

    # transform the attack data to be used for SHAP
    orig_x_test, _, adv_x_test, _ = transform_attack_data(attack_data)

    # initialise the SHAP explainer
    explainer = shap.Explainer(predict_labels, tokenizer)

    # calculate the SHAP values
    orig_shap_values = explainer(orig_x_test, fixed_context=1)
    adv_shap_values = explainer(adv_x_test, fixed_context=1)

    orig_shap_padded = pad_embeddings(orig_shap_values)
    adv_shap_padded = pad_embeddings(adv_shap_values)

    shap_padded = np.concatenate((orig_shap_padded, adv_shap_padded))

    orig_labels = np.zeros(orig_shap_padded.shape[0])
    adv_labels = np.ones(adv_shap_padded.shape[0])

    labels = np.concatenate((orig_labels, adv_labels))

    X, Y = shuffle(shap_padded, labels, random_state=42)

    np.save(params.output_file_path + "shap_vals.npy", X)
    np.save(params.output_file_path + "shap_labels.npy", Y)
