from abc import ABC, abstractmethod


class QAOA(ABC):
    @abstractmethod
    def set_circuit(self):
        pass

    @abstractmethod
    def get_cost(self):
        pass

    @abstractmethod
    def get_statevector(self):
        pass

    @abstractmethod
    def callback(self):
        pass

