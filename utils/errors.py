class QuestionError(Exception):
    """
    Error type related to question generation
    """
    def __init__(self, message: str):
        """
        Setting init params
        :param message: Error message
        """
        super().__init__(message)


class AnswerError(Exception):
    """
    Error type related to answer generation
    """
    def __init__(self, message: str):
        """
        Setting init params
        :param message: Error message
        """
        super().__init__(message)


class DistractorError(Exception):
    """
    Error type related to distractor generation
    """
    def __init__(self, message: str):
        """
        Setting init params
        :param message: Error message
        """
        super().__init__(message)


class PreprocessingError(Exception):
    """
    Error type related to preprocessing
    """
    def __init__(self, message: str):
        """
        Setting init params
        :param message: Error message
        """
        super().__init__(message)
