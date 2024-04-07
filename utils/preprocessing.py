from typing import List

from datasets import load_dataset, Dataset
from transformers import PreTrainedTokenizer, BatchEncoding

from utils.enums import DataType


def question_data(data_type: DataType, tokenizer: PreTrainedTokenizer, with_answer: bool = False) -> Dataset:
    """
    Load and preprocess question data

    :param data_type: Type of data, i.e. training/validation/test
    :param tokenizer: Tokenizer to use
    :param with_answer: Whether an answer is included
    :return: Question data
    """
    data = load_dataset('allenai/sciq', split=data_type.value)
    return data.map(lambda examples: _preprocess_question_data(examples, tokenizer, with_answer=with_answer),
                    batched=True)


def distractor_data(data_type: DataType, tokenizer: PreTrainedTokenizer) -> Dataset:
    """
    Load and preprocess distractor data

    :param data_type: Type of data, i.e. training/validation/test
    :param tokenizer: Tokenizer to use
    :return: Distractor data
    """
    data = load_dataset('allenai/sciq', split=data_type.value)
    return data.map(lambda examples: _preprocess_distractor_data(examples, tokenizer), batched=True)


def prefix_context(context: str) -> str:
    """
    Prefix for question with just context

    :param context: Needed to generate question
    :return: Prefix
    """
    return f'Generate a question from context: {context}'


def prefix_context_answer(context: str, answer: str) -> str:
    """
    Prefix for question with context and answer

    :param context: Needed to generate question
    :param answer: Answer to the question
    :return: Prefix
    """
    return f'Generate a question from context: {context} with answer: {answer}'


def prefix_distractors(question: str, answer: str, context: str) -> str:
    """
    Prefix for distractor with question, answer and context

    :param question: Question needing distractors
    :param context: Needed to generate question
    :param answer: Answer to the question
    :return: Prefix
    """
    return f'Generate 3 distractors for question: {question} with answer: {answer} and context: {context}'


def _preprocess_data(inputs: List[str], targets: List[str] | str, tokenizer: PreTrainedTokenizer) -> BatchEncoding:
    """
    Processes data into specific format

    :param inputs: Input to process
    :param targets: Output target
    :param tokenizer: Tokenizer to use for task
    :return: BatchEncodings of data
    """
    model_inputs = tokenizer(inputs, max_length=128, truncation=True)
    with tokenizer.as_target_tokenizer():
        labels = tokenizer(targets, max_length=32, truncation=True)

    model_inputs['labels'] = labels['input_ids']
    return model_inputs


def _preprocess_question_data(data: Dataset, tokenizer: PreTrainedTokenizer, with_answer: bool = False) -> BatchEncoding:
    """
    Processes data into question format

    :param data: Input to process
    :param tokenizer: Tokenizer to use for task
    :param with_answer: Whether an answer is included or not
    :return: BatchEncodings of question data
    """
    if with_answer:
        inputs = [prefix_context_answer(c, a) for c, a in zip(data['support'], data['correct_answer'])]
    else:
        inputs = [prefix_context(c) for c in data['support']]

    return _preprocess_data(inputs, data['question'], tokenizer)


def _preprocess_distractor_data(data: Dataset, tokenizer: PreTrainedTokenizer) -> BatchEncoding:
    """
    Processes data into distractor format

    :param data: Input to process
    :param tokenizer: Tokenizer to use for task
    :return: BatchEncodings of distractor data
    """
    inputs = [prefix_distractors(q, a, ctx) for q, a, ctx in
              zip(data['question'], data['correct_answer'], data['support'])]
    targets = [f'{d1} <sep> {d2} <sep> {d3}'
               for d1, d2, d3 in zip(data['distractor1'], data['distractor2'], data['distractor3'])]

    return _preprocess_data(inputs, targets, tokenizer)
