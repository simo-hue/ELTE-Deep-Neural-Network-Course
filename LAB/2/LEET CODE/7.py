class NestedIterator(object):
    def __init__(self, nestedList):
        """
        Initialize your data structure here.
        :type nestedList: List[NestedInteger]
        """
        # Inseriamo la lista iniziale al contrario
        self.pila = list(reversed(nestedList))

    def next(self):
        """
        :rtype: int
        """
        # next() ha la garanzia di trovare un intero in cima grazie a hasNext()
        if self.hasNext():
            return self.pila.pop().getInteger()
        raise StopIteration

    def hasNext(self):
        """
        :rtype: bool
        """
        while len(self.pila) > 0:
            # Se la cima è un intero, abbiamo finito
            if self.pila[-1].isInteger():
                return True
            
            # Altrimenti, estraiamo la lista e aggiungiamo gli elementi al contrario
            lst = self.pila.pop().getList()
            for elemento in reversed(lst): 
                self.pila.append(elemento)
            
        return False