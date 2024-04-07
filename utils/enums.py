from enum import Enum


class QuestionModelName(Enum):
    """
    Supported types for question generation models
    """
    POTSAWEE = "potsawee/t5-large-generation-squad-QuestionAnswer"
    T5 = "allenai/t5-small-squad2-question-generation"
    T5FLAN = "google/flan-t5-base"
    FIONA = "kxv26/fiona"
    LESLIE = "kxv26/leslie"


class AnswerModelName(Enum):
    """
    Supported types for answer generation models
    """
    DEEPSET = "deepset/roberta-base-squad2"
    INTEL = "Intel/dynamic_tinybert"
    DISTILIBERT = "distilbert-base-cased-distilled-squad"


class DistractorModelName(Enum):
    """
    Supported types for distractor generation models
    """
    POTSAWEE = "potsawee/t5-large-generation-race-Distractor"
    BART = "voidful/bart-distractor-generation"
    FLAN = "google/flan-t5-base"
    PADO = "kxv26/pado"
    SASO = "kxv26/saso"
    ZANOS = "kxv26/zanos"
    KOLA = "kxv26/kola"


class ModelType(Enum):
    """
    Different model types available
    """
    QUESTION = QuestionModelName
    ANSWER = AnswerModelName
    DISTRACTOR = DistractorModelName


class DataType(Enum):
    """
    Different types of data
    """
    TRAINING = "train"
    VALIDATION = "validation"
