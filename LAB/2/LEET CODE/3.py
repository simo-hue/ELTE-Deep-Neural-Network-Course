nums = [1, 5, 3]
target = 4

def finder_naive(nums, target):
    for i in range(len(nums)):
        res = [i,]
        sum = nums[i]
        for j in range(i + 1, len(nums), 1):
            sum += nums[j]
            res.append(j)
            
            if(sum == target):
                return res
            elif((len(res) == 2)):
                sum = nums[i]
                res = [i,]


def finder_pro(nums, target):
    seen_values = {}
    res = []
    for i in range(len(nums)):
        if nums[i] == target:
            res.append(i)
        
        if((target - nums[i]) in seen_values.keys()):
            return [seen_values[target - nums[i]], i]
        else:
            seen_values[nums[i]] = i
        
    return res
                
print(nums)
print(f"res: {finder_pro(nums, target)}")