from enum import Enum

import numpy as np
from datasets import load_metric
from transformers import T5ForConditionalGeneration, T5Tokenizer, \
    DataCollatorForSeq2Seq, \
    Seq2SeqTrainingArguments, Seq2SeqTrainer, BatchEncoding

from utils.enums import QuestionModelName, DataType, AnswerModelName, DistractorModelName
from utils.preprocessing import question_data, distractor_data
from utils.statistics import compute_rouge1, compute_bleurt

def _compute_rouge1(metric, decoded_preds, decoded_labels):
    result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
    return {key: value.mid.fmeasure for key, value in result.items()}

def _compute_bleurt(metric, decoded_preds, decoded_labels):
    result = metric.compute(predictions=decoded_preds, references=decoded_labels)
    return {key: value[0] for key, value in result.items()}

class MetricType(Enum):
    """
    Different types of metrics
    """

    ROUGE = ('rouge', 'rouge1', _compute_rouge1)
    BLEURT = ('bleurt', 'scores', _compute_bleurt)

    def __new__(cls, *args, **kwds):
        value = len(cls.__members__) + 1
        obj = object.__new__(cls)
        obj._value_ = value
        return obj

    def __init__(self, metric_name, training_arg, get_score):
        self.metric_name = metric_name
        self.training_arg = training_arg
        self.get_score = get_score


class Trainer:
    """
    Class used for training custom models
    """
    def __init__(self, model_name: QuestionModelName | AnswerModelName | DistractorModelName, metric_type: MetricType):
        """
        Setting init params

        :param model_name: Name of HuggingFace model to use
        """
        self.model_name = model_name
        self.metric_type = metric_type
        self.tokenizer = T5Tokenizer.from_pretrained(model_name.value)
        self.metric = load_metric(metric_type.metric_name)

    def train_questions(self, model_name, with_answer: bool = False) -> any:
        """
        Train custom question model

        :param with_answer: With or without an answer provided
        :return: New custom trained question model
        """
        train_inputs = question_data(DataType.TRAINING, self.tokenizer, with_answer=with_answer)
        valid_inputs = question_data(DataType.VALIDATION, self.tokenizer, with_answer=with_answer)

        if with_answer:
            model_name += '-answer'

        return self._train(model_name, train_inputs, valid_inputs)

    def train_distractors(self, model_name):
        """
        Train custom distractor model

        :return: New custom trained distractor model
        """
        train_inputs = distractor_data(DataType.TRAINING, self.tokenizer)
        valid_inputs = distractor_data(DataType.VALIDATION, self.tokenizer)

        return self._train(model_name, train_inputs, valid_inputs)

    def _compute_metrics(self, eval_pred: any) -> any:
        """
        Compute evaluation metrics during training
        :param eval_pred:
        :return: metrix
        """
        preds, labels = eval_pred

        # Replace -100 labels by pad token id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        
        decoded_preds  = self.tokenizer.batch_decode(preds , skip_special_tokens=True)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        return self.metric_type.get_score(self.metric, decoded_preds, decoded_labels)

    def _train(self, name: str, train_inputs: BatchEncoding, valid_inputs: BatchEncoding) -> any:
        """
        General function to train custom model

        :param train_inputs: Input encodings
        :param valid_inputs: Validation encodings
        :return: New custom trained model
        """
        train_args = Seq2SeqTrainingArguments(
            'trained/output',
            evaluation_strategy='steps',
            eval_steps=100,
            logging_strategy='steps',
            logging_steps=100,
            save_strategy='steps',
            save_steps=200,
            learning_rate=4e-5,
            per_device_train_batch_size=8,
            per_device_eval_batch_size=8,
            weight_decay=0.01,
            save_total_limit=3,
            num_train_epochs=1,
            predict_with_generate=True,
            load_best_model_at_end=True,
            metric_for_best_model=self.metric_type.training_arg
        )

        trainer = Seq2SeqTrainer(
            args=train_args,
            model=T5ForConditionalGeneration.from_pretrained(self.model_name.value),
            train_dataset=train_inputs,
            eval_dataset=valid_inputs,
            data_collator=DataCollatorForSeq2Seq(self.tokenizer),
            tokenizer=self.tokenizer,
            compute_metrics=self._compute_metrics
        )

        trainer.train(resume_from_checkpoint=False)
        trainer.save_model(f'trained/{name}')

        return trainer.model
