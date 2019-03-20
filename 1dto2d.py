import numpy as np
from sklearn.preprocessing import scale




## Let print no ignore
# np.set_printoptions(threshold=np.nan)

x = np.load('./dataset5/X1.npy')
# y = np.load('./dataset3/y1.npy')


## normalize to zero to one
x_min = np.min(x,axis = 0)
x_max = np.max(x,axis = 0)
x_nor = (x-x_min)/(x_max-x_min)


# padding
zero = np.zeros((x_nor.shape[0],400))
zero[:x_nor.shape[0],:x_nor.shape[1]] = x_nor
x_nor = zero.reshape(-1,20,20)

print(x_nor.shape)
np.save('./dataset5/X1_2d_nor.npy', x_nor)
# np.save('./y1_2d_nor.npy', y)

# print(x[0])
# print(x_nor[0])