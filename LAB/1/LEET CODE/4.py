def sortino(nums):
    x = nums.copy()
    
    x.sort()

    res = [0] * len(x)

    for i in range(1, (len(x)), 1):
        if(x[i] > x[i - 1]):
            res[i] += res[i - 1] + 1
    
    final = [0] * len(x)
    
    for i in range(len(x)):
        inds = nums.index(nums[i])
        
        
        final[inds] = res[x.index(nums[i])]
        
        print(f"Elemento {nums[i]} ha: {res[inds]} numeri più piccoli")
        
    return final
         
print(sortino([6,5,4,8]))