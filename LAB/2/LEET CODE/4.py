class ParkingSystem(object):

    def __init__(self, big, medium, small):
        """
        :type big: int
        :type medium: int
        :type small: int
        """
        self.big = big
        self.medium = medium
        self.small = small
        

    def addCar(self, carType):
        """
        :type carType: int
        :rtype: bool
        """
        if carType == 1:
            self.big -= 1
            return self.big >= 0
        elif carType == 2:
            self.medium -= 1
            return self.medium >= 0
        elif carType == 3:
            self.small -= 1
            return self.small >= 0

        


# Your ParkingSystem object will be instantiated and called as such:
# obj = ParkingSystem(big, medium, small)
# param_1 = obj.addCar(carType)