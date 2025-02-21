---
layout: post
title: Leetcode
subtitle: 狠狠刷题
date: 2024-09-12
author: PY
header-img: img/git-command.jpg
catalog: true
tags:
  - Leetcode
  - Data Structure
---



# 刷题笔记

## 1 相向双指针

### 两数之和

给你一个下标从 **1** 开始的整数数组 `numbers` ，该数组已按 **非递减顺序排列** ，请你从数组中找出满足相加之和等于目标数 `target` 的两个数。如果设这两个数分别是 `numbers[index1]` 和 `numbers[index2]` ，则 `1 <= index1 < index2 <= numbers.length` 。

以长度为 2 的整数数组 `[index1, index2]` 的形式返回这两个整数的下标 `index1` 和 `index2`。

你可以假设每个输入 **只对应唯一的答案** ，而且你 **不可以** 重复使用相同的元素。

你所设计的解决方案必须只使用常量级的额外空间。

相向双指针模板题目

```python
start,end = 0,len(numbers)-1
        res = []

        while start<end:
            htsum = numbers[start]+numbers[end]
            if htsum == target:
                res.append(start+1)
                res.append(end+1)
                return res
            if htsum < target:
                start+=1
            if htsum > target:
                end-=1
        return []
```

检查双指针的头和尾，如果和大于target，既然是有序的，那么末尾这个加任何一个前面的都大于，那么就尾部减一。如果和小于target，同理，加尾部往前任何一个就会小于，那么就头部指针加一。

### 三数之和

给你一个整数数组 `nums` ，判断是否存在三元组 `[nums[i], nums[j], nums[k]]` 满足 `i != j`、`i != k` 且 `j != k` ，同时还满足 `nums[i] + nums[j] + nums[k] == 0` 。请你返回所有和为 `0` 且不重复的三元组。

**注意：**答案中不可以包含重复的三元组。

这道题可以把nums[i] + nums[j] + nums[k] == 0 改为 nums[j] + nums[k] == -nums[i] 

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        res= []
        n = len(nums)
        for i in range(0,len(nums)-2):      #枚举所有可能的i
            ressub = []
            x = nums[i]
            if i>0 and x==nums[i-1]:
                continue
            j = i+1     #既然加起来是0，那么往后找就行，毕竟是有序的
            k = n-1
            while j<k:
                sm = nums[j]+nums[k]+nums[i]
                if sm<0:
                    j+=1
                if sm>0:
                    k-=1
                if sm==0:
                    res.append([nums[i],nums[j],nums[k]])
                    j+=1
                    while j<k and nums[j] == nums[j-1]:  #去除重复的，很关键，首先多个答案，因此需要把j和k调整，继续遍历
                        j+=1
                    k-=1
                    while j<k and nums[k] == nums[k+1]:  #如果遇到一样的值，那就跳过
                        k-=1
        return res
```

底下部分代码是一样的，但是一开始i用一个for循环枚举可能的i

### 盛水最多的容器

给定一个长度为 `n` 的整数数组 `height` 。有 `n` 条垂线，第 `i` 条线的两个端点是 `(i, 0)` 和 `(i, height[i])` 。

找出其中的两条线，使得它们与 `x` 轴共同构成的容器可以容纳最多的水。

返回容器可以储存的最大水量。

**说明：**你不能倾斜容器。

![image-20240802162006115](img/leetcode/image-20240802162006115.png)

```python
class Solution:
    def maxArea(self, height: List[int]) -> int:
        left,right = 0,len(height)-1
        res = -inf
        while left<right:
            leftbar = height[left]
            rightbar = height[right]
            liq = min(leftbar,rightbar)*(right-left)
            if liq>res:
                res = liq
            if leftbar<=rightbar:
                left+=1
            if leftbar>rightbar:
                right-=1
        return res

```

同样的，相向双指针，总是移动较短的拿一根的指针，因为如果我们选择了目前较短的那一根，我们移动左侧的指针，如果左侧更短，那么左侧的不但见笑了底，还减小了高，很蠢。如果左侧的更长了，那么有效的高度还是右侧固定的这一根，并且底还变短了，也是没有意义的，因此需要移动较短的那一根。

### 接雨水

给定 `n` 个非负整数表示每个宽度为 `1` 的柱子的高度图，计算按此排列的柱子，下雨之后能接多少雨水。

 ![image-20240802172559556](img/leetcode/image-20240802172559556.png)

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        prefix = [0]*len(height)
        postfix = [0]*len(height)
        currentheiget = height[0]
        for i in range(0,len(height)):         #前缀最高
            if height[i]<=currentheiget:
                prefix[i] = currentheiget
            else:
                currentheiget = height[i]
                prefix[i] = currentheiget
        currentheiget = height[-1]
        for i in range(len(height)-1,-1,-1):      #后缀最高
            if height[i]<=currentheiget:
                postfix[i] = currentheiget
            else:
                currentheiget = height[i]
                postfix[i] = currentheiget
        res = 0
        for i in range(len(height)):            #计算每一个小桶，小桶的有效高度是两个板子中较低的有效高度减去本身的高度，也就是底的高度
            realh = min(prefix[i],postfix[i])-height[i]
            res+=realh
        return res
```

我们可以这么思考，当前这个格子看作一个底边长为1的桶，他的左侧板和右侧板为，到目前为止左侧的最高高度和右侧的最高高度，因此使用两个遍历计算出前缀最高和后缀最高，然后一个格子一个格子的计算。





## 2 滑动窗口

### 最长不重复子序列

给定一个字符串 `s` ，请你找出其中不含有重复字符的 最长 子串的长度。

```python
class Solution:
    def lengthOfLongestSubstring(self, s: str) -> int:
        left = right = 0
        res = 0
        while right < len(s):
            if s[right] in s[left:right]:
                left+=1
            else:
                right +=1
            res = max(res,right-left)
        return res
```

只能背了

当右侧的已经出现在结果中的时候就从左边开始缩小，反之就继续伸展右侧的。

### 最短目标子序列

给定一个含有 `n` 个正整数的数组和一个正整数 `target` **。**找出该数组中满足其总和大于等于 `target` 的长度最小的 **子数组**`[numsl, numsl+1, ..., numsr-1, numsr]` ，并返回其长度**。**如果不存在符合条件的子数组，返回 `0` 。

```python
class Solution:
    def minSubArrayLen(self, target: int, nums: List[int]) -> int:
        if not nums:
            return 0
        left = right = 0
        res = float("inf")
        csum = 0
        while right <len(nums):
            csum+=nums[right]
            while csum>=target:
                res = min(res,right-left+1)
                csum-=nums[left]
                left+=1
                
            right+=1
        
        if res ==float("inf"): 
            return 0 
        else :
            return res
```

### ![image-20240802193031463](img/leetcode/image-20240802193031463.png)

```python
class Solution:
    def numSubarrayProductLessThanK(self, nums: List[int], k: int) -> int:
        left,right = 0,0
        currentmul = 1
        res = 0
        while right<len(nums):
            currentmul*=nums[right]
            right+=1
            while currentmul >=k and left < right:
                currentmul/=nums[left]
                left+=1
            res +=right-left        #这是很重要的，因为是要计算你所有字串的个数，如果一个串满足，那么其中的right-left个就都满足
        return res

       


```

左右缩放



## 3 二分查找

这是一个边界条件非常严格的专题，有很多种不同的写法，效果都是一样的

#### 三种写法

```python
# 闭区间写法
def lower_bound(nums: List[int], target: int) -> int:
    left, right = 0, len(nums) - 1  # 闭区间 [left, right]
    while left <= right:  # 区间不为空
        # 循环不变量：
        # nums[left-1] < target
        # nums[right+1] >= target
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1  # 范围缩小到 [mid+1, right]
        else:
            right = mid - 1  # 范围缩小到 [left, mid-1]
    return left

# 左闭右开区间写法
def lower_bound2(nums: List[int], target: int) -> int:
    left = 0
    right = len(nums)  # 左闭右开区间 [left, right)
    while left < right:  # 区间不为空
        # 循环不变量：
        # nums[left-1] < target
        # nums[right] >= target
        mid = (left + right) // 2
        if nums[mid] < target:
            left = mid + 1  # 范围缩小到 [mid+1, right)
        else:
            right = mid  # 范围缩小到 [left, mid)
    return left  # 返回 left 还是 right 都行，因为循环结束后 left == right

# 开区间写法
def lower_bound3(nums: List[int], target: int) -> int:
    left, right = -1, len(nums)  # 开区间 (left, right)
    while left + 1 < right:  # 区间不为空
        mid = (left + right) // 2
        # 循环不变量：
        # nums[left] < target
        # nums[right] >= target
        if nums[mid] < target:
            left = mid  # 范围缩小到 (mid, right)
        else:
            right = mid  # 范围缩小到 (left, mid)
    return right

```

可以仔细查看它们之间的不同。

#### 在排序数组中查找元素的第一个和最后一个位置

![image-20240803140727687](img/leetcode/image-20240803140727687.png)

解决这个问题

```python
class Solution:
    def searchRange(self, nums: List[int], target: int) -> List[int]:
        if not nums:
            return [-1,-1]
        
        def bis(nums,target):
            left,right = 0,len(nums)-1
            while left<=right:
                middle = (left+right)//2
                if nums[middle]<target:
                    left = middle+1
                else:
                    right = middle-1
            return left
        
        start = bis(nums,target)     #这个办法会找到第一个
        if start == len(nums) or nums[start] != target: #这两个判断的顺序不能调换，后面的放到前面就会数组越界
            return [-1,-1]
        end = bis(nums,target+1)-1        #找到比他大1的位置-1就可以找到末尾
        return [start,end]
```

#### 寻找旋转排序数组中的最小值

![image-20240803153927412](img/leetcode/image-20240803153927412.png)

这里我还是坚持使用闭区间，用中间值和数组最左侧的值进行比较

```python
class Solution:
    def findMin(self, nums: List[int]) -> int:
        left = 0
        right = len(nums)-1
        while left<=right:
            middle = (left+right)//2
            middlenumber = nums[middle]
            if middlenumber>nums[-1]:
                left = middle+1
            else:
                right =middle-1
        return nums[left]
            
        
```

非常帅的题

#### 搜索旋转排序数组

![image-20240804152924632](img/leetcode/image-20240804152924632.png)

```python
class Solution:
    def search(self, nums: List[int], target: int) -> int:
        def leftside(i):
            end = nums[-1]  #最后一个
            if nums[i]>end:   #中点在左边一段,也就是排序好的一段
                return target>end and nums[i]>=target   #target和他在同一段
            else:       #中点在右边一段
                return target>end or nums[i]>=target   #在左边一段高的部分   在左边一段低的部分

        left,right =0,len(nums)-1
        while left<=right:
            middle = (left+right)//2
            if leftside(middle):
                right = middle-1
            else:
                left = middle+1
        if left == len(nums) or nums[left]!= target:
            return -1
        else:
            return left
```

只需要额外设计一个函数，判断因该舍弃哪一部分的数组就可以了

## 4.链表

### 反转链表

![image-20240804161228682](img/leetcode/image-20240804161228682.png)

解答

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseList(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return  None
        dummy = ListNode()
        prev = None
        cur = head
        while cur:
            nextcure = cur.next   #先记录后一个节点
            cur.next = prev     #反转
            prev = cur    #更新前一个
            cur = nextcure     #更新当前
        return prev
```

一般来说链表题都会有一个当前节点，和前一个结点，然后以当前节点是否为空进行判断

这里要注意记录当前节点的后一个节点，因为反转后原来的下一个节点就无法通过当前节点的next来获取了

#### 反转链表2

![image-20240804192500031](img/leetcode/image-20240804192500031.png)

非常恶心人的一道题

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseBetween(self, head: Optional[ListNode], left: int, right: int) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        p0 = dummy
        for _ in range(left-1):  #首先找到头节点
            p0 = p0.next
        prev = None
        cur = p0.next
        for _ in range(right-left+1):  #这边开始反转，要注意，这次cur是第一个，所以prev是None
            nexto =cur.next
            cur.next = prev
            prev = cur
            cur = nexto
        p0.next.next= cur           #这个时候p0还是指向的后一个，所以next就可以指向被反转序列的最后一个，然后把他指向不需要反转的后一部分
        p0.next = prev             #指向被反转序列的头部
        return dummy.next

```

#### k个一组反转链表

![image-20240805161448549](img/leetcode/image-20240805161448549.png)



```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def reverseKGroup(self, head: Optional[ListNode], k: int) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        count = 0
        counthead = head
        while counthead:
            counthead = counthead.next
            count+=1
        p0 = dummy
        while count>=k:
            count-=k 
            prev =None
            cur = p0.next      
            for _ in range(k):
                nextone = cur.next
                cur.next = prev
                prev = cur
                cur = nextone
            nextp0 = p0.next
            p0.next.next = cur
            p0.next = prev
            p0 = nextp0    
        return dummy.next    
```

![image-20240805162118986](img/leetcode/image-20240805162118986.png)

做这种题一定要定位p0，cur和prev



### 链表中的快慢指针

#### 链表的中间节点

![image-20240805163316874](img/leetcode/image-20240805163316874.png)

可以使用一个每次走两步的和一个每次走一步的指针，快的到了，慢的也就正好到了一半，但是边缘条件要处理好

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def middleNode(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        slowp = head
        fastp = head
        while fastp:
            if not fastp.next or not fastp:
                return slowp
            else:
                slowp =slowp.next
                fastp= fastp.next.next
        return slowp
```

#### 判断链表是否有环

![image-20240805172056758](img/leetcode/image-20240805172056758.png)

用字典记录是一种办法

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        dic = {}
        dummy = ListNode()
        dummy.next = head
        cur = head
        while cur:
            if cur not in dic:
                dic[cur] = 1
                cur = cur.next         
            else:
                return True
        return False

```

还有一种就是也用快慢指针，因为如果有环，那么快指针一定会追上慢指针

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def hasCycle(self, head: Optional[ListNode]) -> bool:
        dummy = ListNode()
        dummy.next = head
        slow = head
        fast = head
        while fast and fast.next:
            slow = slow.next
            fast = fast.next.next
            if fast == slow:
                return True
        return False

```

#### 环形链表2 找到环出现的点

![image-20240805174125435](img/leetcode/image-20240805174125435.png)

哈希表

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, x):
#         self.val = x
#         self.next = None

class Solution:
    def detectCycle(self, head: Optional[ListNode]) -> Optional[ListNode]:
        cur = head
        dic = {}
        count = 0
        while cur:
            count+=1
            if cur not in dic:
                dic[cur]=0
            else:
                return cur
            cur = cur.next
        return None
```

### 链表中的快慢指针

#### 删除链表的倒数第n个节点

![image-20240806142011301](img/leetcode/image-20240806142011301.png)

一般来说，当有可能会需要删除头节点的时候，使用dummy节点

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def removeNthFromEnd(self, head: Optional[ListNode], n: int) -> Optional[ListNode]:
        dummy = ListNode()
        dummy.next = head
        fast = dummy
        slow = dummy
        for _ in range(n):    #先走
            fast = fast.next
        while fast.next:
            fast = fast.next
            slow = slow.next        #快指针到了，我就到了
        slow.next = slow.next.next  #删除
        return dummy.next
```

快指针先移动n步，然后慢指针开始跑，快指针到头了，慢指针也到了要删除的节点之前了。

#### 删除排序链表中的重复元素

![image-20240806143713938](img/leetcode/image-20240806143713938.png)

如果是排序好的，就可以使用快慢指针。

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        if not head:
            return None
        dummy = ListNode()
        dummy.next = head
        cur = head
        while cur.next:
            vic = cur.next
            if vic.val  == cur.val:
                cur.next = vic.next
            else:
                cur = cur.next
        return head

        # if not head:
        #     return None
        # dummy = ListNode()
        # dummy.next = head
        # fast = dummy
        # middle = dummy
        # tail = dummy
        # fast = fast.next.next
        # middle = middle.next
        # while fast:
        #     if fast.val == middle.val:
        #         print("youle")
        #         tail.next = fast
        #         middle = fast
        #         fast = fast.next
        #     else:    
        #         fast = fast.next
        #         middle = middle.next
        #         tail = tail.next
        # return dummy.next
        

```

#### 删除列表中重读的元素,连带他本身

![image-20240806160435131](img/leetcode/image-20240806160435131.png)

这个可以先用哈希表做，做出来最重要

```python
# Definition for singly-linked list.
# class ListNode:
#     def __init__(self, val=0, next=None):
#         self.val = val
#         self.next = next
class Solution:
    def deleteDuplicates(self, head: Optional[ListNode]) -> Optional[ListNode]:
        dic = {}
        dummy = ListNode()
        dummy.next = head
        cur = head
        while cur:
            if cur.val in dic:
                dic[cur.val]+=1
            else:
                dic[cur.val]=1
            cur = cur.next
        cur = dummy
        while cur.next:
            if dic[cur.next.val]>1:
                print(dic[cur.next.val])
                cur.next = cur.next.next
            else:
                cur = cur.next
        return dummy.next

```

## 5.二叉树

### 递归二叉树

#### 二叉树的最大深度

![image-20240806164408193](img/leetcode/image-20240806164408193.png)

![image-20240806164414620](img/leetcode/image-20240806164414620.png)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def maxDepth(self, root: Optional[TreeNode]) -> int:
        def dfs(root):         #第一种方法
            if not root:
                return 0
            return max(dfs(root.left),dfs(root.right))+1

        res = 0
        def dfs2(root,count):       #第二种方法
            if not root:
                return 
            count+=1
            nonlocal res
            res = max(res,count)
            dfs2(root.left,count)
            dfs2(root.right,count)

        dfs2(root,0)     
        return res
```

#### 相同的树

![image-20240806174356505](img/leetcode/image-20240806174356505.png)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isSameTree(self, p: Optional[TreeNode], q: Optional[TreeNode]) -> bool:
        
        def dfs(p,q):
            if not p and not q:
                return True
            if not p or not q:
                return False
            return p.val == q.val and dfs(p.left,q.left) and dfs(p.right,q.right)
        return dfs(p,q)
```

#### 对称二叉树

![image-20240806175147044](img/leetcode/image-20240806175147044.png)

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right

class Solution:
    def isSymmetric(self, root: Optional[TreeNode]) -> bool:
        left = root.left
        right = root.right
        
        def dfs(left,right):                 #传入一个左边的树和右边的树，因为根节点是分界点，所以拆成左右两棵树
            if not left and not right:
                return True

            if not left or not right:
                return False
            
            return left.val == right.val and dfs(left.left,right.right) and dfs(left.right,right.left)  #递归左边的左子树和右边的右子树，左边的右子树和右边的左子树，判断他们是否一样呢
        return dfs(left,right)
```

#### 平衡二叉树

![image-20240806181418663](img/leetcode/image-20240806181418663.png)

非常好的一道题

```python
# Definition for a binary tree node.
# class TreeNode:
#     def __init__(self, val=0, left=None, right=None):
#         self.val = val
#         self.left = left
#         self.right = right
class Solution:
    def isBalanced(self, root: Optional[TreeNode]) -> bool:

        def dfs(root):
            if not root:
                return 0
            left = dfs(root.left)
            if left == -1:         #如果有-1，那就一直返回
                return -1
            right = dfs(root.right)   #这边也是一样 
            if right == -1 or abs(left-right)>1:      #但是加一个条件，就是不平衡的时候也返回-1
                return -1
            return max(left,right)+1        #当前层高度的计算方法模板max     否则就返回正常的高度
        return dfs(root)!=-1    
            
        
```

#### 二叉树的右视图

非常不错的一道题，可以用层序遍历，也可以使用递归

![image-20240808000450321](img/leetcode/image-20240808000450321.png)

##### 层序遍历

```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        if not root:
            return []
        
        queue = Deque([root])
        res = []
        while queue:
            clevel  =[]
            for i in range(len(queue)):   #标准的层序遍历模板
                cnode = queue.popleft()
                clevel.append(cnode)
                if cnode.left:
                    queue.append(cnode.left)
                if cnode.right:
                    queue.append(cnode.right)
            res.append(clevel[-1].val)    
                
        return res
```

##### 递归

![image-20240808000847654](img/leetcode/image-20240808000847654.png)

```python
class Solution:
    def rightSideView(self, root: Optional[TreeNode]) -> List[int]:
        ans = []

        def dfs(node,depth):    #深度优先遍历
            if not node:
                return
            if depth == len(ans):       #因为会先到每一层的最右侧，如果别的的节点挡住，那么那个节点一定已经在答案里了
                
                ans.append(node.val)
            dfs(node.right,depth+1)     #先把右子树遍历完
            dfs(node.left,depth+1)
        dfs(root,0)
        return ans
```

#### 验证二叉树

这里就需要实现带区间的树递归参数传递

![image-20240809125914441](img/leetcode/image-20240809125914441.png)

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        #太牛逼了

        def dfs(root,leftbound,rightbound):
            if not root:
                return True    #空树就是搜索的，没毛病
            v = root.val        
            ok =  v<rightbound and v >leftbound      #验证一下搜索树的本地性质，现在符合么
            return ok and dfs(root.left,leftbound,v) and dfs(root.right,v,rightbound)   #右子树和左子树也要符合
        #并且，区间需要变化，这里的区间更新可以详见下面一张图，左子树就更新右边界，右子树就更新左边界
        return dfs(root,-inf,inf)
```

![image-20240809130115615](img/leetcode/image-20240809130115615.png)

 第二种方法基于，中序遍历二叉搜索树后会得到一个严格递增的数组

```python
class Solution:
    def isValidBST(self, root: Optional[TreeNode]) -> bool:
        #太牛逼了

        #还有一种方法们就是中序遍历二叉搜索树会得到一个严格递增的数组
        res = []
        def inorder(root):
            if not root:
                return 
            inorder(root.left)
            res.append(root.val)
            inorder(root.right)
        inorder(root)
        for i in range(1,len(res)):
            if res[i]<=res[i-1]:
                return False
        return True
```

#### 二叉树的最近公共祖先

![image-20240809153057799](img/leetcode/image-20240809153057799.png)

分类讨论

如果在递归中找到了p或q，就直接返回就行，因为如果另一个在其下方，那么最近公共祖先都是一开始被找到的那个p或q

![image-20240809153508858](img/leetcode/image-20240809153508858.png)

如果左右子树都找到了，那么返回自身，因为别的情况不可能

![image-20240809153552904](img/leetcode/image-20240809153552904.png)

如果只有左子树找到了，那么但肯定不在右子树中，因此只需要递归左子树的结果就可以

![image-20240809153640078](img/leetcode/image-20240809153640078.png)

因此结果为

![image-20240809153706788](img/leetcode/image-20240809153706788.png)

```python
class Solution:
    def lowestCommonAncestor(self, root: 'TreeNode', p: 'TreeNode', q: 'TreeNode') -> 'TreeNode':

        def lca(root):
            if not root or root == p or root == q:   #这种情况都是返回自身
                return root
            left = lca(root.left)        #看看能不能找到
            right = lca(root.right)
            if left and right:           #都找到了那就ok
                return root
            if left:
                return left
            else:
                return right
        return lca(root)
```



##### 如果这棵树是二叉搜索树

```python
def lca(root):
            x = root.val
            if p.val < x and q.val <x: #都在左边
                return lca(root.left)
            if p.val > x and q.val >x: #都在右边
                return lca(root.right)
            else:
                return root           #空节点或者一个在左一个在右就直接返回就可以了
        return lca(root)
```

二叉树的代码也是同样适用的

### BFS 广度优先搜索，可以使用基于Deque的层序遍历

![image-20240809181906175](img/leetcode/image-20240809181906175.png)

```python
class Solution:
    def levelOrder(self, root: Optional[TreeNode]) -> List[List[int]]:
        if not root:
            return []
        queue = Deque([root]) #先传入第一个节点
        res = []
        while queue:     #这里每次到这里就是一层所有的
            layer = []
            for _ in range(len(queue)): #遍历这一层的每一个
                current = queue.popleft() #从头部开始，这样每一层就是分开的
                layer.append(current.val)
                if current.left:          #加入下一层的
                    queue.append(current.left)
                if current.right:
                    queue.append(current.right)
            res.append(layer)       #为了返回值的形式
        return res           
```



## 6.回溯

#### 电话号码的字母组合

![image-20240809185424619](img/leetcode/image-20240809185424619.png)

```python
class Solution:
    def letterCombinations(self, digits: str) -> List[str]:
        if not digits:
            return []
        res = []
        dic ={
            "2":"abc",
            "3":"def",
            "4":"ghi",
            "5":"jkl",
            "6":"mno",
            "7":"pqrs",
            "8":"tuv",
            "9":"wxyz"
        }
        def dfs(index,path):
            if index == len(digits):     #这里的index指的是到了这个digits的第几位
                res.append("".join(path))     #这里复杂度是n
                return
            current = dic[digits[index]]   #当前位可以选择的字母     这里由于最多可以有4个字母，因此是4^n
            for i in current:
                path.append(i)
                dfs(index+1,path)
                path.pop()     #恢复现场，不然这个path会越来越长
        dfs(0,[])
        return res     
    #时间复杂度O(n*4^n)
```

### 1.子集型回溯

#### 子集

![image-20240813173825448](img/leetcode/image-20240813173825448.png)



```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        res = []
        def dfs(index,path):           #这个index可以转化为操作的次数
            if index == len(nums):
                res.append(path.copy())    #需要copy，不然传入的是一个引用
                return
            dfs(index+1,path)     #不选
            path.append(nums[index])         #这里每次操作数+1
            dfs(index+1,path)      #选
            path.pop()           #恢复现场
        dfs(0,[])
        return res
```

第二种方法

```python
class Solution:
    def subsets(self, nums: List[int]) -> List[List[int]]:
        
        res = []
        n = len(nums)
        def dfs(index,path):       #每次必须选一个
            res.append(path.copy())  #每次递归都加入一下当前的路径
            if index == n:           #这就停止了
                return
            for i in range(index,len(nums)): #每次在剩下的nums中选择
                path.append(nums[i])
                dfs(i+1,path)  #这里是i+1,是我们人为定义的顺序
                path.pop()
        dfs(0,[])
        return res
```



#### 分割回文串

![image-20240813180523648](img/leetcode/image-20240813180523648.png)

```python
class Solution:
    def partition(self, s: str) -> List[List[str]]:
        res = []
        def dfs(index,path):
            if index == len(s):
                res.append(path.copy())  #由于每一个都要出现，因此在最后放入结果
                return
            for i in range(index,len(s)): #j 作为字串的结束部分
                t = s[index:i+1]   #i作为这个字串的结束位置，也就代表了每一个逗号
                if t == t[::-1]:       #简单判断一下，也可以用相向双指针
                    path.append(t)
                    dfs(i+1,path)   #人为规定的顺序
                    path.pop()
        dfs(0,[])
        return res
```



### 2.组合型回溯

#### ![image-20240813210758329](img/leetcode/image-20240813210758329.png)

```python
class Solution:
    def combine(self, n: int, k: int) -> List[List[int]]:
        res = []
        def dfs(index,path):   #经典开局
            if len(path) == k:         #这边可以作为筛选，这里是长度为k的
                res.append(path.copy())
                return
            for i in range(index,n+1):        #在这里使用标准的遍历
                path.append(i)
                dfs(i+1,path)          #确保选择过的不会再选择
                path.pop()
        dfs(1,[])
        return res
```

![image-20240813213803029](img/leetcode/image-20240813213803029.png)



```python
class Solution:
    def combinationSum3(self, k: int, n: int) -> List[List[int]]:
        res = []
        def dfs(index,path,target):
            if target<0:    #剪枝
                return
            if len(path)==k and sum(path)==n:  #组合问题，一样的，在这里加一个判断
                res.append(path.copy())
                return
            for i in range(index,10):
                path.append(i)
                target = target-i          #每次减少target，如果大了，那就不需要再遍历了
                dfs(i+1,path,target)
                target = target+i      #恢复现场
                path.pop()
        dfs(1,[],n)        #目标是n
        return res
```

![image-20240813214738748](img/leetcode/image-20240813214738748.png)

```python
class Solution:
    def generateParenthesis(self, n: int) -> List[str]:

        length = 2*n   #总长度
        res = []
        def dfs(index,left,right,path): #这里记录，已经放了多少个，左括号和右括号各有多少个，当前的括号形态
            if index == length and left == right:       #记录答案的时间，长度够了，并且左右括号相等就可以
                res.append("".join(path))
                return
            if left<n:          #这里最重要的是，左括号可以加到n，而有括号的数量不能大于左括号
                path.append("(")
                dfs(index+1,left+1,right,path)
                path.pop()
            if right<left:  #因此每次添加了左括号，右括号才有机会补充
                path.append(")")
                dfs(index+1,left,right+1,path)
                path.pop()
        dfs(0,0,0,[])
        return res
```



### 3.排列型回溯

#### 全排列

![image-20240813220000520](img/leetcode/image-20240813220000520.png)

```python
class Solution:
    def permute(self, nums: List[int]) -> List[List[int]]:
        res = []
        def dfs(index,path):   #经典开局
            if index == len(nums):         #如果达成，就加入答案
                res.append(path.copy())
                return 
            for i in nums:         #这里只需要多判断一步就可以
                if i not in path:  #如果不在就避免了选择重复
                    path.append(i)
                    dfs(index+1,path)
                    path.pop()
        dfs(0,[])
        return res
```

时间复杂度为O (n*n!)

n乘以n的阶乘，因为可选的字母是越来越少的，因此树的深度和每个节点的子节点也是越来越少的

#### 全排列2

![image-20240813222307034](img/leetcode/image-20240813222307034.png)

```python
class Solution:
    def permuteUnique(self, nums: List[int]) -> List[List[int]]:
        res = []
        pos =[0]*len(nums)

        def dfs(index,path,pos):    #这里需要额外加入一个位置列表，用于去除已经使用过了的位置的数
            if index == len(nums):     #去重，因为会有多种相似的路径
                if path not in res:
                    res.append(path.copy())
                    return 
            for i in range(0,len(nums)):
                if pos[i] ==0:        #这个位置的数字如果没有被使用
                    path.append(nums[i])
                    pos[i]=1            #使用这个位置的数字
                    dfs(index+1,path,pos)
                    pos[i]=0
                    path.pop()
        dfs(0,[],pos)
        return res
        
```

#### N皇后

![image-20240814122231934](img/leetcode/image-20240814122231934.png)

```python
class Solution:
    def solveNQueens(self, n: int) -> int:

        hang = [0]*n     #可以通过四个列表来监控横向竖向斜向
        lie = [0]*n 
        otho = [0]*(2*n-1) #-3-2-10123
        etho = [0]*(2*n-1) #3210123

        res = 0
        reapath = []

        def addqueen(x,y):        #添加一个皇后，并且标注攻击位置
            hang[x] = 1
            lie[y] = 1
            otho[x-y+n-1] = 1
            etho[x+y] = 1
        
        def removequeen(x,y):    #删除皇后，并且清除攻击位置
            hang[x] = 0
            lie[y] = 0
            otho[x-y+n-1] = 0
            etho[x+y] = 0
        
        def attacked(x,y):         #判断一下是否会被攻击
            return hang[x] or lie[y] or otho[x-y+n-1] or etho[x+y]
        
        def dfs(y,path):
            if y ==n:          #这里是当y到达底部的时候，结束递归
                nonlocal res        #这里可以在另一题中使用，用于返回结果的数量
                res+=1
                reapath.append(path.copy()) #这个是路径
                return 
            for x in range(n):   #对于每一行的每一个位置
                if not attacked(x,y):   #如果不被攻击就放置，然后递归下一行
                    addqueen(x,y)
                    path.append("."*(x)+"Q"+"."*(n-1-x))     #这里可以制作题目要求的格式
                    dfs(y+1,path)      #y进入下一行
                    path.pop()
                    removequeen(x,y)
        dfs(0,[])    
        print(reapath)
        return reapath
    
    #时间复杂为O(N^2*N!)
```

## 7动态规划

### 从记忆化搜索到递推

所有的DP都可以看作回溯，但是回溯很慢。因此记录回溯的每一步也就变成了DP

#### 打家劫舍

![image-20240814153920958](img/leetcode/image-20240814153920958.png)

可以先用回溯做

```python
class Solution:
    def rob(self, nums: List[int]) -> int:

        res = -inf
        @cache    #记忆化，可以加速，但是函数的参数必须都是可以哈希的
        def dfs(index,current):
            nonlocal res
            res = max(res,current)   #每次都更新最大值
            if index > len(nums)-1:   #由于涉及到index+1和+2，因此遍历边界需要换
                return
            
            a = dfs(index+1,current)  #第一种情况是当前的不选，选下一个，因此index+1可以选
            b = dfs(index+2,current+nums[index])  #第二种情况是当前的选，然后下一个不可以选，因此index+2，然后current加上当前的钱
            
        dfs(0,0)
        return res
```

```python
class Solution:
    def rob(self, nums: List[int]) -> int:

        n = len(nums)
        cache = [-1]*n   #如果不用装饰器，那就自己实现一个
        def dfs(index):
            if index > n-1:
                return 0
            if cache[index]!=-1:         #本质上也是储存结果
                return cache[index]
            res = max(dfs(index+1),dfs(index+2)+nums[index])
            cache[index]=res
            return res
        return dfs(0)

```

接下来使用dp，将空间复杂度也优化到O(1)

```python
class Solution:
    def rob(self, nums: List[int]) -> int:

        dp = [0]*(len(nums)+2)   #用数组记录每一步的最大价值

        for i in range(0,len(nums)):
            dp[i+2] = max(dp[i]+nums[i],dp[i+1]) #这个递推非常关键，我们是通过前面的得到后面的
        return dp[-1]    #最后的那个就是答案
```

```python
class Solution:
    def rob(self, nums: List[int]) -> int:

        dp = [0]*(len(nums)+2)
        f0 = f1 = 0              #其实一共也就用三个数，因此在数组里面换换就行了
        
        for i in range(0,len(nums)):
            newf = max(f1,f0+nums[i])
            f0 = f1
            f1 = newf
        return f1
```

### 背包问题

#### ![image-20240814164309595](img/leetcode/image-20240814164309595.png)

#### 01背包问题

01背包问题中，每样可以选择装入背包的物品是有限个，并不是无限多个的。

这里给出一个最简单的背包问题

背包的容量为capacity，w数组代表了每一个物品占用的空间，v数组代表了他们的价值

求出背包所能够装的最大价值

```python
def zeronebag(capacity,w,v):
    n = len(w)
    @cache
    def dfs(i,c):   #i是指第i个物品，c是指当前还有多少容量
        if i < 0 :     #遍历完成就退出
            return 0
       	if c<w[i]:        #如果剩余容量不够装，那就跳过，返回不装的情况
            return dfs(i-1,c)
       	return max(dfs(i-1,c),dfs(i-1,c-w[i])+v[i])    #如果够，那么可以选或者不选，都递归一下
   	return dfs(n-1,capacity)
    
```

##### 目标和

![image-20240814172211710](img/leetcode/image-20240814172211710.png)

先来一个暴力解法，时间复杂度很高

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:

        res = []
        def dfs(index,currentsum):
            if currentsum == target and index == -1:
                res.append(1)
            if index <0:
                return 0
            dfs(index-1,currentsum+nums[index]) #加或者减  
            dfs(index-1,currentsum-nums[index])
        dfs(len(nums)-1,0)
        return len(res)
```

也可以使用另一种想法

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:

        #p   选择的加号的数的合
        #s-p   选择的减号的数的合
        #p-(s-p) = t       根据题目有
        #2p = s+t      #化简
        #p = (s+t)/2       #转化为
        target+=sum(nums)
        if target<0 or target%2:  #这里可以提前判断，如果不能被2整除，那就证明永远不可以得到
            return 0
        target //=2
        n = len(nums)
        @cache     #加速，我们python玩家真的太有操作了
        def dfs(index,current):
            if index<0:
                return 1 if current == 0 else 0  #如果current被减为0那证明正好够，因此返回1，表示找到了一个可以使用的结果
            if current < nums[index]:       #选不了，太大了
                return dfs(index-1,current)
            return dfs(index-1,current) + dfs(index-1,current- nums[index])    #选或不选，相加是因为要返回结果数量
        
        return dfs(n-1,target)
```

改成递推

```python
class Solution:
    def findTargetSumWays(self, nums: List[int], target: int) -> int:

        target+=sum(nums)
        if target<0 or target%2:
            return 0
        target //=2
        n = len(nums)

        f = [[0]*(target+1) for _ in range(len(nums)+1)]
        #这里0-target一共有target+1个值，后面也是一样
        f[0][0] = 1

        for i,x in enumerate(nums): 
            for c in range(target+1):
                if c < x:
                    f[i+1][c] = f[i][c]   #这个数没选，target也没变
                else:
                    f[i+1][c] = f[i][c-x] + f[i][c]  #所有的组合加起来
        return f[len(nums)][target]
```



```python
    # 计算数组的总和减去目标值的绝对值，并确定转换成的背包问题的容量
        total_sum = sum(nums)
        s = total_sum - abs(target)
        if s < 0 or s % 2:
            return 0
        half_sum = s // 2  # 背包容量

        num_items = len(nums)
        # 初始化动态规划表格，dp[i][j]表示前i个物品中，能够装满容量为j的背包的方案数
        dp = [[0] * (half_sum + 1) for _ in range(num_items + 1)]
        dp[0][0] = 1  # 初始条件：不选择任何物品，装满容量为0的背包有1种方式

        # 遍历每个物品
        for item_index in range(1, num_items + 1):
            # 遍历每个可能的容量
            for capacity in range(half_sum + 1):
                if capacity < nums[item_index - 1]:
                    dp[item_index][capacity] = dp[item_index - 1][capacity]  # 只能不选当前物品
                else:
                    dp[item_index][capacity] = dp[item_index - 1][capacity] + dp[item_index - 1] [capacity - nums[item_index - 1]]  # 不选当前物品 + 选当前物品

        return dp[num_items][half_sum]
```

#### 完全背包问题

![image-20240814180543143](img/leetcode/image-20240814180543143.png)

##### 零钱兑换

![image-20240814192943190](img/leetcode/image-20240814192943190.png)

首先是最蠢的办法

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:

        res = inf
        @cache
        def dfs(index,still,coinn):         #still就是当前还需要选择多少钱  index是当前在哪种硬币上纠结呢
            if still == 0:    #coinn就是当前选择了多少个硬币
                nonlocal res
                res = min(res,coinn) 
                return
            if index < 0 or still <0:
                return
            dfs(index,still-coins[index],coinn+1) #这里和01背包问题不一样的是，在选择后，index不需要-1，你可以继续选，如果不选，再-1
            dfs(index-1,still,coinn)
        dfs(len(coins)-1,amount,0)
        return res if res!=inf else -1  

```

再来一种直接返回的

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        @cache
        def dfs(index,c):
            if index <0:
                return 0 if c==0 else inf
            if c < coins[index]:
                return dfs(index-1,c)
            return min(dfs(index,c-coins[index])+1,dfs(index-1,c))  #这边返回值直接就是硬币个数，因此需要+1

        res = dfs(len(coins)-1,amount)
        return res if res!=inf else -1
```

递推写法

```python
class Solution:
    def coinChange(self, coins: List[int], amount: int) -> int:
        n = len(coins)
        f = [[inf]*(amount+1) for _ in range(len(coins)+1)] #这边都是要+1的因为总的要迭代的对象个数是这么多
        f[0][0] = 0     #第一个位置要初始化
        for cindex,cvalue in enumerate(coins):
            for am in range(amount+1):
                if am < cvalue:           #如果不够
                    f[cindex+1][am] = f[cindex][am]
                else:                     #其他情况就是选或者不选
                    f[cindex+1][am] = min(f[cindex][am],f[cindex+1][am-cvalue]+1)   #这后面一个+1就对应了完全背包的性质
        res = f[n][amount]
        return res if res < inf else -1
```



### 线性DP

#### 最长公共子序列

![image-20240814204448750](img/leetcode/image-20240814204448750.png)

先来一个回溯解法

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:
        n1 = len(text1)     #这里是长度
        n2 = len(text2)
        @cache
        def dfs(i,j):
            if i<0 or j<0:    #如果index到了，那就返回，表明没有了
                return 0
            if text1[i] == text2[j]:     #如果两个字母相等，那就可以都选
                return dfs(i-1,j-1)+1
            return max(dfs(i-1,j),dfs(i,j-1))  #如果不相等，那么只选一个
        return dfs(n1-1,n2-1)
    
   #时间复杂度O(n1*n2)
```

递推写法

```python
class Solution:
    def longestCommonSubsequence(self, text1: str, text2: str) -> int:

        n = len(text1)
        m = len(text2)

        f = [[0]*(m+1) for _ in range(n+1)]

        for i,x in enumerate(text1):
            for j,y in enumerate(text2):
                if x == y:
                    f[i+1][j+1] = f[i][j]+1
                else:
                    f[i+1][j+1] = max(f[i+1][j],f[i][j+1])
        return f[n][m]
```

#### 编辑距离

![image-20240815133917223](img/leetcode/image-20240815133917223.png)

![image-20240815140304897](img/leetcode/image-20240815140304897.png)

首先是回溯写法

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        if not word1 and not word2:
            return 0
        
        n = len(word1)
        m = len(word2)
        @cache
        def dfs(i,j):      
            if i <0:
                return j+1        #如果有任何一个字符串没了，那么就是要删掉另一个的所有，需要这么多操作
            if j<0:
                return i+1
            if word1[i] == word2[j]:
                return dfs(i-1,j-1)       #如果这两个字母一样，那就不需要操作，直接返回下一部分
            else:           #这里是关键，这三个操作，插入，删除，替换，可以抽象为
                            #再word1中插入，也就是在word2中删除一个
                    		#再word1中删除，那就正常删除一个
                        	#替换，那就意味着两边都减少一个
                return min(dfs(i-1,j)+1,dfs(i,j-1)+1,dfs(i-1,j-1)+1)
        return dfs(n-1,m-1)
```

递推

```python
class Solution:
    def minDistance(self, word1: str, word2: str) -> int:
        n = len(word1)
        m = len(word2)
        dp = [[0]*(m+1) for _ in range(n+1)]
        dp[0] = list(range(m+1)) #边界条件

        for i in range(n):
            dp[i+1][0] = i+1  #边界条件
            for j in range(m):
                if word2[j] == word1[i]:
                    dp[i+1][j+1] = dp[i][j]
                else:
                    dp[i+1][j+1] = min(dp[i+1][j],dp[i][j+1],dp[i][j])+1    #等比翻译
                    
        return dp[n][m]
```

#### 最长递增子序列

![image-20240815161039868](img/leetcode/image-20240815161039868.png)

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int:
        n = len(nums)
        @cache
        def dfs(i):  #i表示当前第i个数字作为整个串的末尾
            res = 0
            for j in range(i): #遍历前面每一个数
                if nums[j]<nums[i]:   #如果遇到小的，那就从那边开始
                    res = max(res,dfs(j))
            return res+1
        ans = 0
        for i in range(n): #枚举每一种末尾         O(N^2)
            ans = max(ans,dfs(i))
        return ans
```

```python
class Solution:
    def lengthOfLIS(self, nums: List[int]) -> int: #dfs改成f数组，递归改成循环
        n = len(nums)
        f = [0]*n     
        for i in range(n):
            for j in range(i):
                if nums[j]<nums[i]:
                    f[i] = max(f[i],f[j])
            f[i]+=1
        return max(f)     
```



### 状态机DP

![image-20240815204004197](img/leetcode/image-20240815204004197.png)



#### 买卖股票的最佳时机，不限制交易次数

![image-20240816193722387](img/leetcode/image-20240816193722387.png)

投机取巧的方法

做T，短线交易

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        duanxian = 0
        for i in range(1,len(prices)):
            if prices[i]>prices[i-1]:    #能挣钱么
                duanxian+= prices[i]-prices[i-1]  #那就挣了
        return duanxian
```

 回溯

![image-20240816202808795](img/leetcode/image-20240816202808795.png)



```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        n = len(prices)
        #倒着思考
        @cache
        def dfs(i,hold):
            if i < 0:  #当i<0天，由于一开始不持有股票
                return -inf if hold else 0
            if hold:
                return max(dfs(i-1,True),dfs(i-1,False)-prices[i])
                #           什么也不做     卖出手里的股票
            return max(dfs(i-1,False),dfs(i-1,True)+prices[i])  #买入股票 或者还是什么都不做
        return dfs(n-1,False)

```

翻译成递推

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:

        n = len(prices)

        f = [[0]*2 for _ in range(n+1)]
        f[0][1] = -inf #我们需要多一个状态用于存储，说明第一天是不可能有股票的
        for i,p in enumerate(prices):
            f[i+1][0] = max(f[i][0],f[i][1]+p)  #后面的全部都+1
            f[i+1][1] = max(f[i][1],f[i][0]-p)
        return f[n][0]
```



#### 如果有冷冻期呢

![image-20240816211415201](img/leetcode/image-20240816211415201.png)

```python
class Solution:
    def maxProfit(self, prices: List[int]) -> int:
        n = len(prices)
        #倒着思考
        @cache
        def dfs(i,hold):
            if i < 0:  #当i<0天，由于一开始不持有股票
                return -inf if hold else 0
            if hold:
                return max(dfs(i-1,True),dfs(i-2,False)-prices[i]) #这边由于有冷冻期，因此需要变成i-2，我只能卖前天买的
                #           什么也不做     卖出手里的股票
            return max(dfs(i-1,False),dfs(i-1,True)+prices[i])  #买入股票 或者还是什么都不做
        return dfs(n-1,False)
```

#### 如果有交易次数限制呢

![image-20240816211902082](img/leetcode/image-20240816211902082.png)

![image-20240816211949383](img/leetcode/image-20240816211949383.png)

这里就需要添加一个参数，表示，到i天结束时，完成了最多j笔交易

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
        n = len(prices)
        #倒着思考
        @cache
        def dfs(i,j,hold): #j代表，到i天结束时，完成了最多j笔交易
            if j <0:
                return -inf       #如果j<0 那就证明这个交易是不合法的
            if i < 0:  #当i<0天，由于一开始不持有股票
                return -inf if hold else 0
            if hold:
                return max(dfs(i-1,j,True),dfs(i-1,j,False)-prices[i])
                #           什么也不做     卖出手里的股票
            return max(dfs(i-1,j,False),dfs(i-1,j-1,True)+prices[i])  #买入股票 或者还是什么都不做
                                             #这里j-1意味着减少了一次交易
        return dfs(n-1,k,False)
```

递推

```python
class Solution:
    def maxProfit(self, k: int, prices: List[int]) -> int:
		n = len(prices)
         f = [[[-inf]*2 for _ in range(k+2)] for _ in range(n+1)]  #三维数组
         for j in range(1,k+2):
             f[0][j][0] = 0
         for i,p in enumerate(prices):
             for j in range(1,k+2):
                 f[i+1][j][0] = max(f[i][j][0],f[i][j-1][1]+p)
                 f[i+1][j][1] = max(f[i][j][1],f[i][j][0]-p)
         return f[n][k+1][0]
```

#### 恰好和至少

![image-20240816213526221](img/leetcode/image-20240816213526221.png)

### 区间DP

#### 最长回文子序列

![image-20240816215411899](img/leetcode/image-20240816215411899.png)

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:
        n = len(s)
        @cache
        def dfs(i,j):
            if i > j:
                return 0    #这是一个空串
            if i==j:         # 长度为1的串就是回文，所以直接返回1就可以了
                return 1
            if s[i] == s[j]:
                return dfs(i+1,j-1)+2       #如果两边都相等，那么久都选，然后结果+2
            return max(dfs(i+1,j),dfs(i,j-1))  #要不然就选左边或者选右边
        return dfs(0,n-1)

```

```python
class Solution:
    def longestPalindromeSubseq(self, s: str) -> int:

        n = len(s)
        f = [[0]*n for _ in range(n)]
        for i in range(n-1,-1,-1):    #倒序遍历
            f[i][i] = 1
            for j in range(i+1,n):  #正序遍历
                if s[i] == s[j]:
                    f[i][j] = f[i+1][j-1]+2
                else:
                    f[i][j] = max(f[i+1][j],f[i][j-1])
        return f[0][n-1]
```

#### 多边形三角剖分的最低得分

![image-20240818165035073](img/leetcode/image-20240818165035073.png)

![image-20240818165050375](img/leetcode/image-20240818165050375.png)

![image-20240818165106360](img/leetcode/image-20240818165106360.png)

```python
class Solution:
    def minScoreTriangulation(self, values: List[int]) -> int:
        n = len(values)
        @cache
        def dfs(i,j):
            if i+1 ==j:
                return 0   #不存在三角形
            res = inf
            for k in range(i+1,j):
                res = min(res,dfs(i,k)+dfs(k,j)+values[i]*values[j]*values[k]) #左边形状，本三角形和右边形状
            return res
        return dfs(0,n-1)

```





### 树形DP

#### 二叉树的直径

![image-20240818173714786](img/leetcode/image-20240818173714786.png)

```python
class Solution:
    def diameterOfBinaryTree(self, root: Optional[TreeNode]) -> int:

    #当前链的长度等于左子树的最长加右子树的最长+2
        res = 0
        @cache
        def dfs(node):
            if node is None:
                return -1
            lenl = dfs(node.left)    #左子树的最大高度
            lenr = dfs(node.right)     #右子树的最大高度
            nonlocal res
            res = max(res,lenl+lenr+2)         #结果在这里更新
            return max(lenl,lenr)+1      #这边可以保证返回最大高度
        dfs(root)
        return res
    
   # O(N)
```

#### 二叉树中的最大路径和

![image-20240818180927770](img/leetcode/image-20240818180927770.png)

```python
class Solution:
    def maxPathSum(self, root: Optional[TreeNode]) -> int:
        res = -inf  #取最大那就用-inf
        
        def dfs(node):
            if not node:
                return 0
            sumnodeleft = dfs(node.left)   #左子树的最大和
            sumnoderight = dfs(node.right) #右子树的最大和
            nonlocal res
            res = max(res,sumnodeleft+sumnoderight+node.val)       #左子树的最大和+右子树的最大和
            return max(0,max(sumnodeleft,sumnoderight)+node.val)    #保证返回左子树或者右子树的最大值加上当前节点值
            #这边和0比大小很关键，因为如果算出一个部分是负数，那还不如不选，因此要和0比大小，这里的0不是指结点的值为0而是指子树的整体大小为0
        dfs(root)
        return res

```



#### 一般树中，相邻字符不同的最长路径

![image-20240818184335866](img/leetcode/image-20240818184335866.png)

```python
class Solution:
    def longestPath(self, parent: List[int], s: str) -> int:
        n = len(parent)
        g = [[] for  _ in range(n)] #这边也可以用哈希表
        for i in range(1,n):        #把每一个位置的子节点做好
            g[parent[i]].append(i)
        
        ans = 0

        def dfs(x):
            nonlocal ans
            lenofx = 0 # 初始化变量lenofx，表示从当前节点x向下延伸的最长路径
            for y in g[x]:       #对于每一个节点的子节点
                lenofy = dfs(y)+1    # 递归调用dfs计算子节点y的最长路径，并加1表示包括当前节点y
                if s[y]!=s[x]:    # 只有当子节点y的字符和当前节点x的字符不同时，才考虑更新路径
                    ans = max(ans,lenofx+lenofy)   # 更新全局的最长路径长度ans
                    lenofx = max(lenofx,lenofy)    # 更新当前节点x的最长路径长度
            return lenofx     # 返回从当前节点x向下延伸的最长路径
        dfs(0)
        return ans+1
```

#### 打家劫舍3

![image-20240818190212782](img/leetcode/image-20240818190212782.png)

```python
class Solution:
    def rob(self, root: Optional[TreeNode]) -> int:

        res = 0  # 这个变量`res`其实没有用到，可以移除

        def dfs(node):
            if not node:
                return 0, 0  # 如果当前节点为空，返回(0, 0)，表示两种情况都为0
            
            # 递归计算左子树的两种情况的收益
            lrob, lnotrob = dfs(node.left)
            # 递归计算右子树的两种情况的收益
            rrob, rnotrob = dfs(node.right)
            
            # 抢劫当前节点的收益：当前节点的值 + 左右子树不能抢劫子节点的最大收益
            rob = lnotrob + rnotrob + node.val
            # 不抢劫当前节点的收益：左右子树的两种情况的最大值之和
            notrob = max(lrob, lnotrob) + max(rrob, rnotrob) #左右子树可以随便选，因为我们选择了他
            
            # 返回当前节点的两种情况的收益
            return rob, notrob

        # 对根节点调用dfs，并返回两种情况的最大值
        return max(dfs(root))
```

#### 监控二叉树

![image-20240818193728388](img/leetcode/image-20240818193728388.png)

如果根节点不装，那么至少他有一个子节点要装

![image-20240818194103264](img/leetcode/image-20240818194103264.png)



```python
class Solution:
    def minCameraCover(self, root: Optional[TreeNode]) -> int:
        def dfs(node):
            if node is None:
                return inf, 0, 0  # 空节点的三种状态：不能安装摄像头，0个摄像头即可满足被监控到，不用被监控
            
            l_choose, l_by_fa, l_by_children = dfs(node.left)  # 处理左子节点
            r_choose, r_by_fa, r_by_children = dfs(node.right)  # 处理右子节点
            
            # 当前节点选择放置摄像头的情况
            choose = min(l_choose, l_by_fa) + min(r_choose, r_by_fa) + 1
            
            # 当前节点不放摄像头，而由父节点来监控的情况
            by_fa = min(l_choose, l_by_children) + min(r_choose, r_by_children)
            
            # 当前节点不放摄像头，由子节点来监控的情况
            by_children = min(l_choose + r_by_children, l_by_children + r_choose, l_choose + r_choose)
            
            return choose, by_fa, by_children
        
        choose, _, by_children = dfs(root)  # 根节点没有父节点，所以只考虑选择摄像头或由子节点监控
        return min(choose, by_children)

```













### 迷宫DP

地下城游戏

![image-20240818123931311](img/leetcode/image-20240818123931311.png)



```python
class Solution:
    def calculateMinimumHP(self, dungeon: List[List[int]]) -> int:
        n, m = len(dungeon), len(dungeon[0])
        BIG = 10**9
        dp = [[BIG] * (m + 1) for _ in range(n + 1)]
        dp[n][m - 1] = dp[n - 1][m] = 1
        for i in range(n - 1, -1, -1):
            for j in range(m - 1, -1, -1):
                minn = min(dp[i + 1][j], dp[i][j + 1])
                dp[i][j] = max(minn - dungeon[i][j], 1)

        return dp[0][0]
```

### 

## 单调栈

#### 每日温度

![image-20240818214621097](img/leetcode/image-20240818214621097.png)

```python
class Solution:
    def dailyTemperatures(self, temperatures: List[int]) -> List[int]:
        #O(min(n,U))    U = temperatures[MAX] - temperatures[MIN]+1
        n = len(temperatures)
        res = [0]*n
        stack = []
        for i in range(n-1,-1,-1): #倒着遍历
            t = temperatures[i]       #当前的温度
            while stack and temperatures[stack[-1]]<=t: #对于栈中小于自身的，全部pop掉，因为他们没用
                                                     #为什么呢，因为之后遍历到的，最高肯定不会是那些比自己小的，而是自己
                stack.pop()
            if stack: #如果栈不为空，那么栈顶数据比自己大，可以作为答案
                res[i] = stack[-1]-i #索引，距离当前有多少天
            stack.append(i)    #每次都会把当前索引放进去
        return res
```

#### 接雨水

![image-20240819145959426](img/leetcode/image-20240819145959426.png)

单调栈做法

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        ans = 0  # 用于存储最后的接雨水总量
        st = []  # 这是一个栈，用于存储柱子的索引
        
        for i, h in enumerate(height):
            # 栈非空且当前高度大于或等于栈顶元素的高度时，说明可以形成凹槽
            while st and h >= height[st[-1]]:
                bottom_h = height[st.pop()]  # 弹出栈顶元素，表示凹槽的底部高度
                
                if len(st) == 0:
                    break  # 如果栈为空，说明没有左边的边界，不能形成凹槽，跳出循环
                
                left = st[-1]  # 栈顶元素弹出后，现在的栈顶就是左边界
                dh = min(height[left], h) - bottom_h  # 计算水的高度，取左边界和当前高度的较小值减去底部高度
                
                # 接水量为水的高度乘以宽度，宽度为 `i - left - 1`
                ans += dh * (i - left - 1)
            
            st.append(i)  # 将当前柱子的索引入栈
        
        return ans  # 返回最终接住的雨水总量
```

最大前缀做法

```python
class Solution:
    def trap(self, height: List[int]) -> int:
        # 初始化前缀和后缀数组
        prefix = [0] * len(height)
        postfix = [0] * len(height)
        # 计算每个位置的前缀最大高度
        currentheiget = height[0]
        for i in range(0, len(height)):
            if height[i] <= currentheiget:
                prefix[i] = currentheiget
            else:
                currentheiget = height[i]
                prefix[i] = currentheiget
        # 计算每个位置的后缀最大高度
        currentheiget = height[-1]
        for i in range(len(height) - 1, -1, -1):
            if height[i] <= currentheiget:
                postfix[i] = currentheiget
            else:
                currentheiget = height[i]
                postfix[i] = currentheiget
        # 计算能够存储的水量
        res = 0
        for i in range(len(height)):
            realh = min(prefix[i], postfix[i]) - height[i]
            res += realh

        return res

```

## 单调队列

#### 滑动窗口最大值

![image-20240819154650874](img/leetcode/image-20240819154650874.png)

```python
class Solution:
    def maxSlidingWindow(self, nums: List[int], k: int) -> List[int]:
        ans = []  # 用来存放结果，即每个滑动窗口的最大值
        q = deque()  # 双端队列，存储的是元素的索引
        
        for i, x in enumerate(nums):
            # 1. 入窗口：保证队列中的元素按照从大到小的顺序排列
            while q and nums[q[-1]] <= x:
                q.pop()  # 队列中如果存在比当前元素小的元素，它们永远不可能成为最大值，移除它们
            q.append(i)  # 将当前元素的索引加入到队列中
            
            # 2. 出窗口：检查队首元素是否还在当前窗口中，如果不在，则移出队列
            if i - q[0] >= k:
                q.popleft()  # 队首元素已经不在窗口范围内了，移出队列
            
            # 3. 记录答案：当遍历到第 `k` 个元素时开始记录窗口最大值
            if i >= k - 1:
                ans.append(nums[q[0]])  # 队首元素就是当前窗口的最大值
        
        return ans  # 返回所有滑动窗口的最大值

```






