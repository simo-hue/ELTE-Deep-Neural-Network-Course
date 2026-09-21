class MinStack(object):

    def __init__(self):
        self.min = []
        self.v = []
        

    def push(self, value):
        """
        :type value: int
        :rtype: None
        """
        if isinstance(value, int):
            self.v.append(value)

            if len(self.min) == 0:
                self.min.append(value)
            else:
                if value < self.min[-1]:
                    self.min.append(value)
                else:
                    self.min.append(self.min[-1])

    def pop(self):
        """
        :rtype: None
        """
        if len(self.v) > 0:
            self.v.pop()
        if len(self.min) > 0:
            self.min.pop()

    def top(self):
        """
        :rtype: int
        """
        return self.v[-1]

    def getMin(self):
        """
        :rtype: int
        """
        return self.min[-1]
    
    def __str__(self):
        return f"STACK: {self.v} e MIN: {self.min}"
        
obj = MinStack()
print(obj)
obj.push(1)
print(obj)
obj.push(2)
print(obj)
print(obj.top())
print(obj.getMin())
print(obj.pop())
print(obj.getMin())
print(obj.top())