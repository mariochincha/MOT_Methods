# trackers/base_tracker.py

from abc import ABC, abstractmethod

class BaseTracker(ABC):

    @abstractmethod
    def update(self, detections):
        pass