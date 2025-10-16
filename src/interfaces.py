from abc import abstractmethod, ABC


class ICrackMaskExtractor(ABC):

    @abstractmethod
    def extract(self):
        pass
