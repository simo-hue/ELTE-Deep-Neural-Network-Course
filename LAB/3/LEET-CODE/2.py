import numpy as np

def solution(candies, extraCandies):
    """
        :type candies: List[int]
        :type extraCandies: int
        :rtype: List[bool]
        """
    if(len(candies) < 1):
        return False
        
    return [(np.array(candies) + extraCandies) >= np.max(candies)]
    
    
l = [2,3,5,1,3]
o = np.array([True,True,True,False,True])

print(solution(l, 3))
print(o)