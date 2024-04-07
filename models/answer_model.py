from transformers import (AutoModelForQuestionAnswering, AutoTokenizer, pipeline,
                          DistilBertTokenizer, DistilBertForQuestionAnswering)

from utils.enums import AnswerModelName
from utils.errors import AnswerError


class AnswerModel:
    """
    Model used to generate answers to questions
    """
    def __init__(self, model_name: AnswerModelName):
        """
        Initialize models

        :param model_name: Name of HuggingFace answer generator model to use
        """
        model_name = model_name.value
        match model_name:
            case AnswerModelName.DEEPSET | AnswerModelName.INTEL:
                model = AutoModelForQuestionAnswering.from_pretrained(model_name)
                tokenizer = AutoTokenizer.from_pretrained(model_name)
            case AnswerModelName.DISTILIBERT:
                model = DistilBertForQuestionAnswering.from_pretrained(model_name)
                tokenizer = DistilBertTokenizer.from_pretrained(model_name)
            case _:
                # Use DEEPSET as default
                model = AutoModelForQuestionAnswering.from_pretrained(AnswerModelName.DEEPSET.value)
                tokenizer = AutoTokenizer.from_pretrained(AnswerModelName.DEEPSET.value)

        self.nlp = pipeline('question-answering', model=model, tokenizer=tokenizer)

    def generate(self, question: str, context: str) -> str:
        """
        Uses the provided question and context to generate a correct answer to the question

        :param question: The question to generate an answer from
        :param context: Background information needed for generation
        :return: Generated answer
        """
        answer = self.nlp(question=question, context=context)

        if not answer:
            raise AnswerError("Distractors could not be generated.")
        return answer.get("answer")
