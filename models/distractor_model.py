from transformers import AutoTokenizer, AutoModelForSeq2SeqLM, T5ForConditionalGeneration, T5Tokenizer

from utils.enums import DistractorModelName


class DistractorModel:
    """
    Model used to generate distractors for a question that a similar to the correct answer
    """
    def __init__(self, model_name: DistractorModelName):
        """
        Initialize models

        :param model_name: Name of HuggingFace answer generator model to use
        """
        self.model_name = model_name
        match model_name:
            case DistractorModelName.POTSAWEE | DistractorModelName.BART:
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_name.value)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(self.model_name.value)
            case DistractorModelName.PADO | DistractorModelName.SASO | DistractorModelName.ZANOS:
                self.model = T5ForConditionalGeneration.from_pretrained(self.model_name.value)
                self.tokenizer = T5Tokenizer.from_pretrained(self.model_name.value)
                self.tokenizer.sep_token = "<sep>"

    def generate(self, answer: str, context: str, question: str = None):
        """
        Uses provided question, answer and context to generate distractors

        :param question: Question for which distractors are needed
        :param answer: If an answer is provided, this will not be generated and used for question generation
        :param context: Background information needed for generation
        :return: Generated distractors
        """
        if question is None:
            question = ""
        if self.model_name is DistractorModelName.BART:
            input_text = " ".join([context, self.tokenizer.sep_token, question, self.tokenizer.sep_token, answer])
        elif self.model_name is DistractorModelName.KOLA:
            input_text = " ".join([answer, self.tokenizer.sep_token, context])
        else:
            input_text = " ".join([question, self.tokenizer.sep_token, answer, self.tokenizer.sep_token, context])

        inputs = self.tokenizer(input_text, return_tensors="pt")
        outputs = self.model.generate(**inputs, max_new_tokens=128)
        if outputs is None:
            return None
        distractors = self.tokenizer.decode(outputs[0], skip_special_tokens=False)
        distractors = distractors.replace(self.tokenizer.pad_token, "").replace(self.tokenizer.eos_token, "")
        distractors = [y.strip() for y in distractors.split(self.tokenizer.sep_token)]

        if not distractors:
            return None
        return distractors
