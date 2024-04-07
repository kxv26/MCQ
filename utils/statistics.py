import json
import logging
import datasets
import evaluate
import logging
import time
import pandas as pd
import numpy as np
from models.answer_model import *
from models.question_model import *
from models.distractor_model import *
from models.pipelines import *

logging.basicConfig(level=logging.INFO)


def load_test_data():
    """
    Load the SciQ test dataset.

    :return: the SciQ test dataset
    """
    data = datasets.load_dataset('allenai/sciq', split='test')
    return data


def save_scores_question_models(dataset, score_field, filename, description=""):
    """
    Save the scores from a dataset to a JSON file.

    :param dataset: dataset containing the scores
    :param score_field: field containing the scores
    :param filename: name of the file to save the scores to
    :param description: description of the scores
    :return: None
    """
    scores_only = []

    for i in range(len(dataset)):
        entry = dataset[i]
        score_value = entry[score_field] if score_field in entry else None
        scores_only.append(score_value)

    data_to_save = {
        "description": description,
        "scores": scores_only
    }

    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data_to_save, file, ensure_ascii=False, indent=4)

    logging.info(f"{score_field} data successfully saved to {filename}.")


def compare_question_models():
    """
    Compare the performance of the Potsawee and AllenAI question models with our trained models on the SciQ test
    dataset.

    :return: None
    """
    data_test = load_test_data()

    potsawee_question = QuestionModel(QuestionModelName.POTSAWEE)
    allenai_question = QuestionModel(QuestionModelName.T5)
    fine_tuned_question = QuestionModel(QuestionModelName.FIONA)

    scorer_bleu = evaluate.load('bleu')
    scorer_rouge = evaluate.load('rouge')
    scorer_bleurt = evaluate.load("bleurt", 'bleurt-large-512')
    scorer_bertscore = evaluate.load("bertscore")

    def score_head(entry, question_model, model_name, scorer, score_name, score_arg1):
        generated_q = question_model.generate(entry['support'])
        if model_name == 'custom2':
            generated_q = question_model.generate(entry['support'], entry['correct_answer'])
        if not generated_q or not entry['question']:
            return {f'{model_name}_{score_name}': None}
        if generated_q is None or entry['question'] is None:
            return {f'{model_name}_{score_name}': None}

        if score_name == 'bleurt':
            score = scorer.compute(predictions=[generated_q], references=[entry['question']])
        elif score_name == 'bertscore':
            score = scorer.compute(predictions=[generated_q], references=[entry['question']], lang='en')
        else:
            score = scorer.compute(predictions=[generated_q], references=[[entry['question']]])

        if score_name == 'bleurt':
            return {f'{model_name}_{score_name}': score['scores'][0]}
        if score_name == 'bertscore':
            return {f'{model_name}_{score_name}': score['precision'][0]}
        return {f'{model_name}_{score_name}': score[score_arg1]}

    def score_general(entry, question_model, model_name, scorer, score_name):
        return score_head(entry, question_model, model_name, scorer, score_name, score_name)

    def score_rouge(entry, question_model, model_name, score):
        return score_head(entry, question_model, model_name, scorer_rouge, score, score)

    def score_potsawee_bleu(entry):
        return score_general(entry, potsawee_question, 'potsawee', scorer_bleu, 'bleu')

    def score_allenai_bleu(entry):
        return score_general(entry, allenai_question, 'allenai', scorer_bleu, 'bleu')

    def score_custom_bleu(entry):
        return score_general(entry, fine_tuned_question, 'custom', scorer_bleu, 'bleu')

    def score_custom_with_answer_bleu(entry):
        return score_general(entry, fine_tuned_question, 'custom2', scorer_bleu, 'bleu')

    def score_potsawee_rouge1(entry):
        return score_rouge(entry, potsawee_question, 'potsawee', 'rouge1')

    def score_allenai_rouge1(entry):
        return score_rouge(entry, allenai_question, 'allenai', 'rouge1')

    def score_custom_rouge1(entry):
        return score_rouge(entry, fine_tuned_question, 'custom', 'rouge1')

    def score_custom_with_answer_rouge1(entry):
        return score_rouge(entry, fine_tuned_question, 'custom2', 'rouge1')

    def score_potsawee_rouge2(entry):
        return score_rouge(entry, potsawee_question, 'potsawee', 'rouge2')

    def score_allenai_rouge2(entry):
        return score_rouge(entry, allenai_question, 'allenai', 'rouge2')

    def score_custom_rouge2(entry):
        return score_rouge(entry, fine_tuned_question, 'custom', 'rouge2')

    def score_custom_with_answer_rouge2(entry):
        return score_rouge(entry, fine_tuned_question, 'custom2', 'rouge2')

    def score_potsawee_rougeL(entry):
        return score_rouge(entry, potsawee_question, 'potsawee', 'rougeL')

    def score_allenai_rougeL(entry):
        return score_rouge(entry, allenai_question, 'allenai', 'rougeL')

    def score_custom_rougeL(entry):
        return score_rouge(entry, fine_tuned_question, 'custom', 'rougeL')

    def score_custom_with_answer_rougeL(entry):
        return score_rouge(entry, fine_tuned_question, 'custom2', 'rougeL')

    def score_potsawee_bleurt(entry):
        return score_general(entry, potsawee_question, 'potsawee', scorer_bleurt, 'bleurt')

    def score_allenai_bleurt(entry):
        return score_general(entry, allenai_question, 'allenai', scorer_bleurt, 'bleurt')

    def score_custom_bleurt(entry):
        return score_general(entry, fine_tuned_question, 'custom', scorer_bleurt, 'bleurt')

    def score_custom_with_answer_bleurt(entry):
        return score_general(entry, fine_tuned_question, 'custom2', scorer_bleurt, 'bleurt')

    def score_potsawee_bertscore(entry):
        return score_general(entry, potsawee_question, 'potsawee', scorer_bertscore, 'bertscore')

    def score_allenai_bertscore(entry):
        return score_general(entry, allenai_question, 'allenai', scorer_bertscore, 'bertscore')

    def score_custom_bertscore(entry):
        return score_general(entry, fine_tuned_question, 'custom', scorer_bertscore, 'bertscore')

    def score_custom_with_answer_bertscore(entry):
        return score_general(entry, fine_tuned_question, 'custom2', scorer_bertscore, 'bertscore')

    save_scores_question_models(data_test.map(score_potsawee_bleu, batched=False),
                                "potsawee_bleu",
                                "./question_models_results/potsawee_score_bleu.json",
                                "Potsawee BLEU scores")

    save_scores_question_models(data_test.map(score_allenai_bleu, batched=False),
                                "allenai_bleu",
                                "./question_models_results/allenai_score_bleu.json",
                                "Allenai BLEU scores")

    save_scores_question_models(data_test.map(score_custom_bleu, batched=False),
                                "custom_bleu",
                                "./question_models_results/custom_score_bleu.json",
                                "Custom BLEU scores")

    save_scores_question_models(data_test.map(score_custom_with_answer_bleu, batched=False),
                                "custom2_bleu",
                                "./question_models_results/custom_with_answer_score_bleu.json",
                                "Custom with answer BLEU scores")

    save_scores_question_models(data_test.map(score_potsawee_rouge1, batched=False),
                                "potsawee_rouge1",
                                "./question_models_results/potsawee_score_rouge1.json",
                                "Potsawee ROUGE-1 scores")

    save_scores_question_models(data_test.map(score_allenai_rouge1, batched=False),
                                "allenai_rouge1",
                                "./question_models_results/allenai_score_rouge1.json",
                                "Allenai ROUGE-1 scores")

    save_scores_question_models(data_test.map(score_custom_rouge1, batched=False),
                                "custom_rouge1",
                                "./question_models_results/custom_score_rouge1.json",
                                "Custom ROUGE-1 scores")

    save_scores_question_models(data_test.map(score_custom_with_answer_rouge1, batched=False),
                                "custom2_rouge1",
                                "./question_models_results/custom_with_answer_score_rouge1.json",
                                "Custom with answer ROUGE-1 scores")

    save_scores_question_models(data_test.map(score_potsawee_rouge2, batched=False),
                                "potsawee_rouge2",
                                "./question_models_results/potsawee_score_rouge2.json",
                                "Potsawee ROUGE-2 scores")

    save_scores_question_models(data_test.map(score_allenai_rouge2, batched=False),
                                "allenai_rouge2",
                                "./question_models_results/allenai_score_rouge2.json",
                                "Allenai ROUGE-2 scores")

    save_scores_question_models(data_test.map(score_custom_rouge2, batched=False),
                                "custom_rouge2",
                                "./question_models_results/custom_score_rouge2.json",
                                "Custom ROUGE-2 scores")

    save_scores_question_models(data_test.map(score_custom_with_answer_rouge2, batched=False),
                                "custom2_rouge2",
                                "./question_models_results/custom_with_answer_score_rouge2.json",
                                "Custom with answer ROUGE-2 scores")

    save_scores_question_models(data_test.map(score_potsawee_rougeL, batched=False),
                                "potsawee_rougeL",
                                "./question_models_results/potsawee_score_rougeL.json",
                                "Potsawee ROUGE-L scores")

    save_scores_question_models(data_test.map(score_allenai_rougeL, batched=False),
                                "allenai_rougeL",
                                "./question_models_results/allenai_score_rougeL.json",
                                "Allenai ROUGE-L scores")

    save_scores_question_models(data_test.map(score_custom_rougeL, batched=False),
                                "custom_rougeL",
                                "./question_models_results/custom_score_rougeL.json",
                                "Custom ROUGE-L scores")

    save_scores_question_models(data_test.map(score_custom_with_answer_rougeL, batched=False),
                                "custom2_rougeL",
                                "./question_models_results/custom_with_answer_score_rougeL.json",
                                "Custom with answer ROUGE-L scores")

    save_scores_question_models(data_test.map(score_potsawee_bleurt, batched=False),
                                "potsawee_bleurt",
                                "./question_models_results/potsawee_score_bleurt.json",
                                "Potsawee BLEURT scores")

    save_scores_question_models(data_test.map(score_allenai_bleurt, batched=False),
                                "allenai_bleurt",
                                "./question_models_results/allenai_score_bleurt.json",
                                "Allenai BLEURT scores")

    save_scores_question_models(data_test.map(score_custom_bleurt, batched=False),
                                "custom_bleurt",
                                "./question_models_results/custom_score_bleurt.json",
                                "Custom BLEURT scores")

    save_scores_question_models(data_test.map(score_custom_with_answer_bleurt, batched=False),
                                "custom2_bleurt",
                                "./question_models_results/custom_with_answer_score_bleurt.json",
                                "Custom with answer BLEURT scores")

    save_scores_question_models(data_test.map(score_potsawee_bertscore, batched=False),
                                "potsawee_bertscore",
                                "./question_models_results/potsawee_score_bertscore.json",
                                "Potsawee BERTScore scores")

    save_scores_question_models(data_test.map(score_allenai_bertscore, batched=False),
                                "allenai_bertscore",
                                "./question_models_results/allenai_score_bertscore.json",
                                "Allenai BERTScore scores")

    save_scores_question_models(data_test.map(score_custom_bertscore, batched=False),
                                "custom_bertscore",
                                "./question_models_results/custom_score_bertscore.json",
                                "Custom BERTScore scores")

    save_scores_question_models(data_test.map(score_custom_with_answer_bertscore, batched=False),
                                "custom2_bertscore",
                                "./question_models_results/custom_with_answer_score_bertscore.json",
                                "Custom with answer BERTScore scores")


def save_answering_model_data_to_json(data, filename):
    """
    Save data to a JSON file.

    :param data: Data to be saved.
    :param filename: Filename for the saved JSON.
    :return: None
    """
    with open(filename, 'w', encoding='utf-8') as file:
        json.dump(data, file, ensure_ascii=False, indent=4)
    logging.info(f"Data successfully saved to {filename}.")


def compare_answering_model():
    """
    Compare the performance of the answering models on the SciQ test dataset.

    :return: None
    """
    data_test = load_test_data()

    answer_models = {
        'deepset': AnswerModel(AnswerModelName.DEEPSET),
        'intel': AnswerModel(AnswerModelName.INTEL),
        'distilbert': AnswerModel(AnswerModelName.DISTILIBERT)
    }

    filenames = {
        'deepset': './answer_models_results/generated_data_deepset.json',
        'intel': './answer_models_results/generated_data_intel.json',
        'distilbert': './answer_models_results/generated_data_distilbert.json'
    }

    generated_data = {name: [] for name in list(answer_models.keys())}

    for entry in data_test:
        support_text = entry['support']
        question = entry['question']
        # Support can be empty.
        if support_text is None or support_text == "":
            for name, model in answer_models.items():
                generated_data[name].append({
                    'question': None,
                    'answer': None
                })
            continue
        # Model can fail to generate question!
        for name, model in answer_models.items():
            answer = model.generate(question, support_text)
            generated_data[name].append({
                'question': question,
                'answer': answer
            })

    for name, data in generated_data.items():
        save_answering_model_data_to_json(data, filenames[name])


def compute_rouge1(metric, decoded_preds, decoded_labels):
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {key: value.mid.fmeasure for key, value in result.items()}


def compute_bleurt(metric, decoded_preds, decoded_labels):
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {key: value[0] for key, value in result.items()}


def compute_pairwise_bleurt_max(scorer_bleurt, decoded_preds, decoded_labels):
    """
    Compute the BLEURT score for a list of predictions and labels.

    :param scorer_bleurt: BLEURT scorer
    :param decoded_preds: ordered list of predictions that yielded the highest BLEURT score
    :param decoded_labels: ordered list of labels that yielded the highest BLEURT score
    """
    logging.info(f"Predictions: {decoded_preds}, labels: {decoded_labels}")
    logging.info(f"Length of predictions: {len(decoded_preds)}, length of labels: {len(decoded_labels)}")
    if not (len(decoded_preds) == len(decoded_labels)):
        logging.error("Length of predictions and labels must be equal.")
        return None, None
    score = 0
    labels = []
    for pred in decoded_preds:
        result = scorer_bleurt.compute(predictions=[pred] * len(decoded_labels), references=decoded_labels)
        max_index = np.argmax(result['scores'])
        score += result['scores'][max_index]
        labels.append(decoded_labels.pop(max_index))
    logging.info(f"Score: {score}, labels: {labels}")
    return score, labels


def generate_and_judge_distractors(model, data_test):
    """
    Generate distractors for the SciQ test dataset and judge them using BLEURT.

    :param model: Distractor model
    :param data_test: SciQ test dataset
    :return: DataFrame containing the generated distractors and BLEURT scores with the ordered true distractors
    """
    scorer_bleurt = evaluate.load('bleurt', 'bleurt-large-512')
    data = pd.DataFrame(
        columns=['correct_answer', 'support', 'generated_distractors', 'bleurt_score',
                 'ordered_true_distractors'])

    for i, example in enumerate(data_test):
        if model.model_name == DistractorModelName.KOLA:
            distractors = model.generate(example['correct_answer'], example['support'], None)
        else:
            distractors = model.generate(example['correct_answer'], example['support'], example['question'])
        logging.info(f"Generated distractors {distractors} for example {i}.")

        example_distractors = []
        if example['distractor1'] is not None:
            example_distractors.append(example['distractor1'])
            logging.info(f"Example distractor {example['distractor1']} for example {i}.")
        if example['distractor2'] is not None:
            example_distractors.append(example['distractor2'])
            logging.info(f"Example distractor {example['distractor2']} for example {i}.")
        if example['distractor3'] is not None:
            example_distractors.append(example['distractor3'])
            logging.info(f"Example distractor {example['distractor3']} for example {i}.")

        logging.info(f"Example distractors {example_distractors} for example {i}.")
        if (None not in distractors) and ('' not in distractors) and len(distractors) == 3:
            metric, labels = compute_pairwise_bleurt_max(scorer_bleurt, distractors, example_distractors)
            logging.info(f"Computed BLEURT score {metric} for example {i} with label order {labels}.")

            if metric is None:
                logging.error(f"Failed to compute BLEURT score for example {i}.")
                data.loc[i] = [example['correct_answer'], example['support'], distractors, None, example_distractors]
                continue
            data.loc[i] = [example['correct_answer'], example['support'], distractors, metric, labels]
        else:
            logging.info(f"Failed to generate distractors for example {i} with generated distractors {distractors}.")
            data.loc[i] = [example['correct_answer'], example['support'], distractors, None, example_distractors]
    return data


def compare_distractor_models():
    """
    Compare the performance of the distractor models on the SciQ test dataset.

    :return: None
    """
    models = {
        'potsawee': DistractorModel(DistractorModelName.POTSAWEE),
        'bart': DistractorModel(DistractorModelName.BART),
        'custom1': DistractorModel(DistractorModelName.PADO),
        'custom2': DistractorModel(DistractorModelName.SASO),
        'custom3': DistractorModel(DistractorModelName.ZANOS),
        'custom4': DistractorModel(DistractorModelName.KOLA)}

    data_test = load_test_data()
    for model in models:
        logging.info(f"Generating distractors for {model}.")
        data = generate_and_judge_distractors(models[model], data_test)
        data.to_json(f'./distractor_models_results/generated_distractors_{model}.json')
        print(
            f"Generated distractors for {model} saved to ./distractor_models_results/generated_distractors_{model}.json")


def compare_pipelines():
    pipe1 = Pipeline1()
    pipe2 = Pipeline2()
    pipe3 = Pipeline3()
    data = load_test_data()

    data_pipe1 = pd.DataFrame(columns=['correct_answer', 'support', 'generated_question', 'generated_distractors', 'time'])
    data_pipe2 = pd.DataFrame(columns=['correct_answer', 'support', 'generated_question', 'generated_distractors', 'time'])
    data_pipe3 = pd.DataFrame(columns=['support', 'generated_question', 'generated_answer', 'generated_distractors', 'time'])
    for i, example in enumerate(data):
        if example['support'] is None or example['support'] == "":
            #data_pipe1.loc[i] = [None, None, None, None, None]
            data_pipe2.loc[i] = [None, None, None, None, None]
            data_pipe3.loc[i] = [None, None, None, None, None]
            continue
        start = time.time()
        q, a, d = pipe1.run(example['support'], example['correct_answer'])
        end1 = time.time() - start
        logging.info(f"Generated question {q} and distractors {d} for example {i}.")
        data_pipe1.loc[i] = [example['correct_answer'], example['support'], q, d, end1]

        start = time.time()
        q, a, d = pipe2.run(example['support'], example['correct_answer'])
        end2 = time.time() - start
        logging.info(f"Generated question {q} and distractors {d} for example {i}.")
        data_pipe2.loc[i] = [example['correct_answer'], example['support'], q, d, end2]

        start = time.time()
        q, a, d = pipe3.run(example['support'])
        end3 = time.time() - start
        logging.info(f"Generated question {q}, answer {a} and distractors {d} for example {i}.")
        data_pipe3.loc[i] = [example['support'], q, a, d, end3]
    data_pipe1.to_json('./pipeline_results/pipeline1_result.json')
    data_pipe2.to_json('./pipeline_results/pipeline2_result.json')
    data_pipe3.to_json('./pipeline_results/pipeline3_result.json')


def load_pipeline_data(name):
    data = pd.read_json(f'./pipeline_results/{name}_result.json')
    return pd.DataFrame(data)


def compare_q(scorers, question, generated_question):
    question_bleu = scorers['bleu'].compute(predictions=[generated_question], references=[question])
    question_rouge = scorers['rouge'].compute(predictions=[generated_question], references=[question])
    question_bleurt = scorers['bleurt'].compute(predictions=[generated_question], references=[question])
    question_bertscore = scorers['bertscore'].compute(predictions=[generated_question], references=[question], lang='en')
    return question_bleu['bleu'], question_rouge['rouge1'], question_rouge['rouge2'], question_rouge['rougeL'], question_bleurt['scores'][0], question_bertscore['precision'][0]


def compare_a(scorers, answer, generated_answer):
    answer_bleu = scorers['bleu'].compute(predictions=[generated_answer], references=[answer])
    answer_rouge = scorers['rouge'].compute(predictions=[generated_answer], references=[answer])
    answer_bertscore = scorers['bertscore'].compute(predictions=[generated_answer], references=[answer], lang='en')
    return answer_bleu['bleu'], answer_rouge['rouge1'], answer_rouge['rouge2'], answer_rouge['rougeL'], answer_bertscore['precision'][0]


def compare_pipeline_results():

    pipeline1 = load_pipeline_data('pipeline1')
    pipeline2 = load_pipeline_data('pipeline2')
    pipeline3 = load_pipeline_data('pipeline3')

    test_data = load_test_data()

    analysis_pipeline1 = pd.DataFrame(columns=['generated_question', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'bleurt', 'bertscore', 'generated_distractors', 'max_bleurt', 'ordered_distractors'])
    analysis_pipeline2 = pd.DataFrame(columns=['generated_question', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'bleurt', 'bertscore', 'generated_distractors', 'max_bleurt', 'ordered_distractors'])
    analysis_pipeline3 = pd.DataFrame(columns=['generated_question', 'bleu', 'rouge1', 'rouge2', 'rougeL', 'bleurt', 'bertscore', 'generated_answer', 'bleu_ans', 'rouge1_ans', 'rouge2_ans', 'rougeL_ans', 'bertscore_ans', 'generated_distractors', 'max_bleurt', 'ordered_distractors'])

    scorers = {
        'bleu': evaluate.load('bleu'),
        'rouge': evaluate.load('rouge'),
        'bleurt': evaluate.load('bleurt', 'bleurt-large-512'),
        'bertscore': evaluate.load('bertscore')
    }

    for i, data_point in enumerate(test_data):
        logging.info(f"Comparing data point {i}.")
        distractors = [data_point['distractor1'], data_point['distractor2'], data_point['distractor3']]
        logging.info(f"Distractors {distractors} for data point {i}.")

        logging.info('Pipeline 1')
        logging.info('Judge generated question')
        generated_question1 = pipeline1.loc[i]['generated_question']
        if generated_question1 is not None:
            bleu, rouge1, rouge2, rougeL, bleurt, bertscore = compare_q(scorers, data_point['question'], generated_question1)
        else:
            bleu, rouge1, rouge2, rougeL, bleurt, bertscore = None, None, None, None, None, None

        logging.info('Judge generated distractors')
        generated_distractors1 = pipeline1.loc[i]['generated_distractors']
        if generated_distractors1 is not None:
            bleurt_distractors1, ordered_distractors1 = compute_pairwise_bleurt_max(scorers['bleurt'], generated_distractors1, distractors.copy())
        else:
            bleurt_distractors1, ordered_distractors1 = None, None
        analysis_pipeline1.loc[i] = [generated_question1, bleu, rouge1, rouge2, rougeL, bleurt, bertscore, generated_distractors1, bleurt_distractors1, ordered_distractors1]

        logging.info('Pipeline 2')
        logging.info('Judge generated question')
        generated_question2 = pipeline2.loc[i]['generated_question']
        if generated_question2 is not None:
            bleu, rouge1, rouge2, rougeL, bleurt, bertscore = compare_q(scorers, data_point['question'], generated_question2)
        else:
            bleu, rouge1, rouge2, rougeL, bleurt, bertscore = None, None, None, None, None, None
        logging.info(f"Judge generated distractors: {pipeline2.loc[i]['generated_distractors']} with correct distractors {distractors}")
        generated_distractors2 = pipeline2.loc[i]['generated_distractors']
        if generated_distractors2 is not None:
            bleurt_distractors2, ordered_distractors2 = compute_pairwise_bleurt_max(scorers['bleurt'], generated_distractors2, distractors.copy())
            logging.info(f'BLEURT distractors: {bleurt_distractors2}, ordered distractors: {ordered_distractors2}')
        else:
            bleurt_distractors2, ordered_distractors2 = None, None
            logging.info(f'Could not judge BLEURT distractors: {bleurt_distractors2}, ordered distractors: {ordered_distractors2}')
        analysis_pipeline2.loc[i] = [generated_question2, bleu, rouge1, rouge2, rougeL, bleurt, bertscore, generated_distractors2, bleurt_distractors2, ordered_distractors2]

        logging.info('Pipeline 3')
        logging.info('Judge generated question')
        generated_question3 = pipeline3.loc[i]['generated_question']
        if generated_question3 is not None:
            bleu, rouge1, rouge2, rougeL, bleurt, bertscore = compare_q(scorers, data_point['question'], generated_question3)
        else:
            bleu, rouge1, rouge2, rougeL, bleurt, bertscore = None, None, None, None, None, None

        logging.info('Judge generated answer')
        generated_answer3 = pipeline3.loc[i]['generated_answer']
        if generated_answer3 is not None:
            bleu_ans, rouge1_ans, rouge2_ans, rougeL_ans, bertscore_ans = compare_a(scorers, data_point['correct_answer'], generated_answer3)
        else:
            bleu_ans, rouge1_ans, rouge2_ans, rougeL_ans, bertscore_ans = None, None, None, None, None

        logging.info(f"Judge generated distractors: {pipeline3.loc[i]['generated_distractors']} with correct distractors {distractors}")
        generated_distractors3 = pipeline3.loc[i]['generated_distractors']
        if generated_distractors3 is not None:
            bleurt_distractors3, ordered_distractors3 = compute_pairwise_bleurt_max(scorers['bleurt'], generated_distractors3, distractors.copy())
            logging.info(f'BLEURT distractors: {bleurt_distractors3}, ordered distractors: {ordered_distractors3}')
        else:
            bleurt_distractors3, ordered_distractors3 = None, None
            logging.info(f'Could not judge BLEURT distractors: {bleurt_distractors3}, ordered distractors: {ordered_distractors3}')
        analysis_pipeline3.loc[i] = [generated_question3, bleu, rouge1, rouge2, rougeL, bleurt, bertscore, generated_answer3, bleu_ans, rouge1_ans, rouge2_ans, rougeL_ans, bertscore_ans, generated_distractors3, bleurt_distractors3, ordered_distractors3]

    analysis_pipeline1.to_json('./pipeline_results/analysis_pipeline1.json')
    analysis_pipeline2.to_json('./pipeline_results/analysis_pipeline2.json')
    analysis_pipeline3.to_json('./pipeline_results/analysis_pipeline3.json')
