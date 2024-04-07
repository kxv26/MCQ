import threading
import logging
from typing import Tuple, List

from models.answer_model import AnswerModel
from models.distractor_model import DistractorModel
from models.question_model import QuestionModel
from utils.enums import QuestionModelName, AnswerModelName, DistractorModelName

logging.basicConfig(level=logging.INFO)


class BaseModel:
    """
    Base model allowing the generation of questions, distractors and potentially answers for the creation
    of scientific multiple choice quizzes
    """

    def __init__(self,
                 question_model_name: QuestionModelName,
                 answer_model_name: AnswerModelName,
                 distractor_model_name: DistractorModelName):
        """
        Init models
        :param question_model_name: Name of question model to use
        :param answer_model_name: Name of answer model to use
        :param distractor_model_name: Name of distractor model to use
        """
        self.question_model = QuestionModel(question_model_name)
        self.answer_model = AnswerModel(answer_model_name)
        self.distractor_model = DistractorModel(distractor_model_name)

    def generate(self, context: str, answer: str = None, parallel: bool = False) -> Tuple[str, str, List[str]]:
        """
        Generate complete question with correct and distractor answers

        :param context: Background information needed for generation
        :param answer: If an answer is provided, this will not be generated and used for question generation
        :param parallel: Whether the question is generated in parallel or not
        :return: Generated question, answer and distractor answers
        """
        # If parallel is enabled and an answer is provided, start the optimal pipeline
        if parallel and answer:
            threads = []
            results = []

            question_func = lambda: self.question_model.generate(context, answer=answer)
            distractor_func = lambda: self.distractor_model.generate(answer, context, None)

            target_functions = [question_func, distractor_func]

            for target_function in target_functions:
                thread = threading.Thread(target=lambda t=target_function: results.append(t()))
                threads.append(thread)

            for thread in threads:
                thread.start()

            for thread in threads:
                thread.join()

            if isinstance(results[0], str):
                question = results[0]
                distractors = results[1]
            else:
                question = results[1]
                distractors = results[0]
            logging.info('Parallel generation complete')
            return question, answer, distractors

        # Else generate in sequence
        question = self.question_model.generate(context, answer=answer)
        logging.info(f'Question generated: {question}')
        if question is None:
            return None, None, None

        if not answer:
            answer = self.answer_model.generate(question, context)
        logging.info(f'Answer either generated or provided as: {answer}')

        if answer is None:
            return None, None, None

        distractors = self.distractor_model.generate(answer, context, question)
        logging.info(f'Distractors generated as: {distractors}')
        return question, answer, distractors
