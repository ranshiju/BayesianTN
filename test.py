import torch as tc
from torch import nn
import time
from collections.abc import Iterable


x = tc.randn(8, 12)
# x = nn.Softmax(dim=0)(x)
norm = x.norm(p=2, dim=0)
x = tc.einsum('ab,b->ab', [x, 1/(norm+1e-10)])
print(x.norm(dim=0))

# y = nn.Softmax(dim=0)(tc.randn(2, ))
y = tc.randn((4, ))
y /= y.norm()
# z = nn.Softmax(dim=0)(tc.randn(2, ))
z = tc.randn((3, ))
z /= z.norm()
print(y.norm(), z.norm())
x = x.view(8, 4, 3)
x1 = tc.einsum('abc,b,c->a', [x, y, z])
print(x1.norm())

