import torch
import torch
import torch.nn as nn

"""
For the inputs of nn.bilinear, the first dimension
of inputs variables in ∗ is usually the “batch” (except in Attention) and is often refered
to as the “batch dimension”, indicating the number of samples stored in this tensor. 

The second dimension in the inputs is often referred to as the “feature dimension”. The extra
“out features =: co” dimension can be viewed as a stack of co matrices, for example, if
the weight tensor A has “out features = 2”, then A = [A(1), A(2)], and the bilinear inner
product is computed as [x⊤
1 A(1)x2 + b, x⊤
1 A(2)x2 + b]. Answer the following questions:
"""

m = nn.Bilinear(20, 30, 40)
input1 = torch.randn(128, 20)
input2 = torch.randn(128, 30)
output = m(input1, input2)
print(output.size())
print(output.shape)


# another_m = nn.Bilinear(2, 3, 3)
# input1 = torch.FloatTensor([0.5,-1,0.5])
# input2 = torch.FloatTensor([0.5,0.5,0.5])
# output = another_m(input1, input2)
# print(output.size())

