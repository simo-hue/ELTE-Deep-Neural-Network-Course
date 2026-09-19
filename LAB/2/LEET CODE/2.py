nums = [99, 105, 2,7,11,15]
target = 9

def finder(nums, target):
    x = sorted(nums)
    res = []
    
    if(len(nums) < 1):
        if(nums[0] == target):
            return [0,]
        else:
            return []        
    
    if(x[len(x) // 2] < target):
        for i in range(len(x) // 2 + 1, len(x) // 2, 1):
            print(f"{x[i]}eee")
    elif(x[len(x) // 2] > target):
        for i in range(len(x) // 2 + 1, len(x) // 2, 1):
            print(f"{x[i]}eee")
    else:
        return len(x) // 2
        
        
    
    
        
        
    
print(nums)
finder(nums, target)