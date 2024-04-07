from models.base_model import BaseModel
from utils.enums import QuestionModelName, AnswerModelName, DistractorModelName


class Pipeline1:
    """
    Pipeline1: Custom Question Model, BART Distractor Model
    This pipeline generates a question and distractors sequentially.
    """
    def __init__(self):
        self.model = BaseModel(QuestionModelName.FIONA, AnswerModelName.DEEPSET, DistractorModelName.POTSAWEE)

    def run(self, context, correct_answer):
        """
        Generate question and distractors

        :param context: Background information needed for generation
        :param correct_answer: Correct answer to the question
        """
        if context is None or context == "":
            return None, None, None
        q, a, d = self.model.generate(context, correct_answer, parallel=False)
        return q, a, d


class Pipeline2:
    """
    Pipeline2: Custom Question Model, BART Distractor Model
    This pipeline generates a question and distractors in parallel.
    """
    def __init__(self):
        self.model = BaseModel(QuestionModelName.FIONA, AnswerModelName.DEEPSET, DistractorModelName.POTSAWEE)

    def run(self, context, correct_answer):
        """
        Generate question and distractors in parallel

        :param context: Background information needed for generation
        :param correct_answer: Correct answer to the question
        """
        if context is None or context == "":
            return None, None, None
        q, a, d = self.model.generate(context, correct_answer, parallel=True)
        return q, a, d


class Pipeline3:
    """
    Pipeline3: Custom Question Model, Deepset Answer Model, BART Distractor Model
    This pipeline generates a question, an answer and distractors sequentially.
    """
    def __init__(self):
        self.model = BaseModel(QuestionModelName.FIONA, AnswerModelName.DEEPSET, DistractorModelName.POTSAWEE)

    def run(self, context):
        """
        Generate question, answer and distractors from a context

        :param context: Background information needed for generation
        """
        if context is None or context == "":
            return None, None, None
        q, a, d = self.model.generate(context, parallel=False)
        return q, a, d
