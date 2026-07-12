from abc import ABC, abstractmethod

class BaseModel(ABC):
    """Abstract Base Class defining the interface for all RailGuard AI models."""

    @abstractmethod
    def train(self, train_data, val_data=None, **kwargs):
        """Trains the model on the provided datasets."""
        pass

    @abstractmethod
    def predict(self, x, **kwargs):
        """Runs model inference on input features or raw images."""
        pass

    @abstractmethod
    def evaluate(self, test_data, **kwargs):
        """Evaluates the model and returns performance metrics."""
        pass

    @abstractmethod
    def save(self, filepath):
        """Saves model weights/parameters to disk."""
        pass

    @abstractmethod
    def load(self, filepath):
        """Loads model weights/parameters from disk."""
        pass
