def isAnagram(a, b):
        if(len(a) != len(b)):
            return False
        
        counter_a = {}
        for char in list(a):
            if char in counter_a.keys():
                counter_a[char] += 1
            else:
                counter_a[char] = 1
                
        counter_b = {}
        for char in list(b):
            if char in counter_b.keys():
                counter_b[char] += 1
            else:
                counter_b[char] = 1
                
        return counter_a == counter_b
        
def caller(strs):
    res = []

    if len(strs) < 1:
        return [[""]]
    elif len(strs) == 1:
        return [[strs]]

    for i in range(len(strs) - 1):
        if(strs[i] in res):
            break
        
        partial = [strs[i]]
            
        for j in range(i + 1, len(strs), 1):
            if(isAnagram(strs[i], strs[j]) and strs[j] not in res):
                partial.append(strs[j])

        res.append(partial)

    return res
 
           
a = ["eat","tea","tan","ate","nat","bat"]
b = [""]
c = ["a"]

print(f"{caller(a)}")
print(f"{caller(b)}")
print(f"{caller(c)}")