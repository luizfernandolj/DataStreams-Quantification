from abc import ABC, abstractmethod

class DriftDetector(ABC):
    
    @abstractmethod
    def Increment(self, new):
        pass

    @abstractmethod
    def Update(self, index):
        pass
    
    @abstractmethod
    def Test(self):
        pass