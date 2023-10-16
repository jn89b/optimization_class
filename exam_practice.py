import torch

# arr1 = torch.arange(10).reshape(5,-1)
# print(arr1)
# print(arr1[::2])
# arr2 = 1*(arr1[::2]) 
# print(arr2)


# def yoyo(arr):
#     k = 0
#     K = torch.max(arr)
#     print("K is: ", K)
#     c = torch.zeros(K+1)
#     while k != K:
#         # print("sum is: ", torch.sum(arr == k))
#         c[k] = torch.sum(arr == k)
#         k += 1
#     print("argmax is: ", torch.argmax(c))
#     print("c is: ", c )
#     return torch.argmax(c)

# yoyo(torch.tensor([1,3,3,5,5,5,3,3,1]))

# def why(n, p):
#     arr = torch.zeros(n // p) #creats aan array of zeros 
#     arr[::2] = 1
#     print(arr)
#     arr[1::2] = -1
#     print(arr)
#     return arr

# print(42//4)
# why(42, 4)


X = torch.tensor([[1,0], [0,1], [1,1], [0,0]])
Y = X[-3:] 
#print(Y)
# print(torch.matmul(Y, Y.T))
print(Y**2)
print(torch.sum(Y**2, axis=1)) #axis=1 means sum along the rows
print(torch.sum(Y**2, axis=1).reshape(-1,1))
Z = torch.sum(Y**2, axis=1).reshape(-1,1) + torch.matmul(Y, Y.T)
print(Z)
