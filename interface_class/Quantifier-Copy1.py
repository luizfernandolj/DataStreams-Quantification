from abc import ABC, abstractmethod

class DriftDetector(ABC):
    @abstractmethod
    def (self, X_train, y_train):
        pass

    @abstractmethod
    def predict(self, X_test):
        pass