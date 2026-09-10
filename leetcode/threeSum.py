def threeSum(nums):
    n = len(nums)
    nums = sorted(nums)
    print(nums)
    res = []
    for i in range(n):
        if i > 0 and nums[i - 1] == nums[i]:
            continue
        target = -nums[i]
        left = i + 1
        right = n - 1
        while (left < right):
            if nums[left] + nums[right] == target:
                res.append([nums[i], nums[left], nums[right]])
                while (left < right and nums[left] == nums[left + 1]):
                    left += 1
                while (left < right and nums[right] == nums[right - 1]):
                    right -= 1
                left += 1
                right -= 1
            elif nums[left] + nums[right] < target:
                left += 1
            elif nums[left] + nums[right] > target:
                right -= 1

    return res


nums = [-1, 0, 1, 2, -1, -4]
res = threeSum(nums=nums)
print(res)