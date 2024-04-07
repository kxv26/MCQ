from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer

from utils.enums import QuestionModelName
from utils.preprocessing import prefix_context, prefix_context_answer


class QuestionModel:
    """
    Model used to generate questions based on a context and potential answer
    """
    def __init__(self, model_name: QuestionModelName):
        """
        Set init params

        :param model_name: Name of HuggingFace answer generator model to use
        """
        self.model_name = model_name

    def generate(self, context: str, answer: str = None) -> str:
        """
        Uses the provided context and potentially an answer to generate a relevant question

        :param context: Background information needed for generation
        :param answer: If an answer is provided, this will not be generated and used for question generation
        :return: Generated question
        """
        match self.model_name:
            case QuestionModelName.POTSAWEE:
                question = self._generate_potsawee_question(context)
            case QuestionModelName.T5 | QuestionModelName.T5FLAN:
                question = self._generate_t5_question(context)
            case QuestionModelName.FIONA | QuestionModelName.LESLIE:
                question = self._generate_custom_question(context, answer=answer)
            case _:
                question = self._generate_potsawee_question(context)

        if not question:
            return ""
        return question

    def _generate_potsawee_question(self, context: str) -> str:
        """
        Uses the provided context to generate a relevant question using the potsawee model

        :param context: Background information needed for generation
        :return: Generated question
        """
        tokenizer = AutoTokenizer.from_pretrained(self.model_name.value)
        model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name.value)
        inputs = tokenizer(context, return_tensors="pt", max_length=1024, truncation=True)
        outputs = model.generate(**inputs, max_length=100)
        question_answer = tokenizer.decode(outputs[0], skip_special_tokens=False)
        question_answer = question_answer.replace(tokenizer.pad_token, "").replace(
            tokenizer.eos_token, "")

        if tokenizer.sep_token in question_answer:
            generated_question, answer = question_answer.split(tokenizer.sep_token)
        else:
            generated_question = question_answer

        return generated_question.strip()

    def _generate_t5_question(self, context: str, **generator_args) -> str:
        """
        Uses the provided context to generate a relevant question using the T5 model

        :param context: Background information needed for generation
        :return: Generated question
        """
        tokenizer = T5Tokenizer.from_pretrained(self.model_name.value)
        model = T5ForConditionalGeneration.from_pretrained(self.model_name.value)
        input_ids = tokenizer.encode(context, return_tensors="pt", max_length=1024, truncation=True)
        res = model.generate(input_ids)
        output = tokenizer.batch_decode(res, skip_special_tokens=True)
        return output[0]

    def _generate_custom_question(self, context: str, answer: str, **generator_args) -> str:
        """
        Uses the provided context to generate a relevant question using the custom T5 model

        :param context: Background information needed for generation
        :param answer: (OPTIONAL) Answer for the question
        :return: Generated question
        """
        if answer:
            model_name = QuestionModelName.LESLIE.value
            inp = prefix_context_answer(context, answer)
        else:
            model_name = QuestionModelName.FIONA.value
            inp = prefix_context(context)
        model = T5ForConditionalGeneration.from_pretrained(model_name)
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        input_ids = tokenizer.encode(inp, return_tensors="pt", max_length=1024, truncation=True)
        res = model.generate(input_ids, **generator_args, max_new_tokens=20)
        output = tokenizer.batch_decode(res, skip_special_tokens=True)
        return output[0]
