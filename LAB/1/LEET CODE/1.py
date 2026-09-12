class Solution(object):
    def runningSum(self, nums):
        """
        :type nums: List[int]
        :rtype: List[int]
        """
        counter = []
        sum = 0

        if(len(nums) < 1):
            return 0

        counter = [nums[]]

        for elem in nums:
            sum += elem
            counter.append(sum)

        return counter