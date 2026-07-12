class IPLPredictorException(Exception):
    """Base exception for all IPL score predictor errors."""
    def __init__(self, message, status_code=400):
        super().__init__(message)
        self.message = message
        self.status_code = status_code


class ModelConfigurationError(IPLPredictorException):
    """Raised when there is an issue loading the Keras model, encoders, or scaler configurations."""
    def __init__(self, message):
        super().__init__(message, status_code=500)


class InvalidMatchConfigError(IPLPredictorException):
    """Raised when match configurations (teams, venues) are invalid or identical."""
    def __init__(self, message):
        super().__init__(message, status_code=400)


class InvalidInningsProgressionError(IPLPredictorException):
    """Raised when innings state progression violates logical constraints (e.g. decreasing runs or wickets)."""
    def __init__(self, message):
        super().__init__(message, status_code=400)
