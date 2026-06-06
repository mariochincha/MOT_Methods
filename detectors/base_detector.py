# detectors/base_detector.py

from abc import ABC, abstractmethod

class BaseDetector(ABC):

    @abstractmethod
    def detect(self, frame):
        pass