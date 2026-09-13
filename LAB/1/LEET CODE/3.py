def kidsWithCandies(candies, extraCandies):
        """
        :type candies: List[int]
        :type extraCandies: int
        :rtype: List[bool]
        """
        if(len(candies) < 0):
            return False
        
        max = candies[0]

        # I found the max element
        for elem in candies:
            if(elem > max):
                max = elem

        # Retur the result
        res = [False] * len(candies)
        
        print(f"Vettore: {res}")

        for i in range(len(candies)):
            if(candies[i] >= max):
                res[i] = True

        return res
    
rt = kidsWithCandies([2,3,5,1,3], 3)

print(rt)