from abc import ABC, abstractmethod


class Embedder(ABC):
    @abstractmethod
    def get_embedding(self, text: str) -> list[float]:
        pass