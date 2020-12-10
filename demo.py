from autodiffcst import AD as ad
from autodiffcst import admath as admath
import numpy as np

if __name__ == "__main__":
    # x = ad.AD(5, tags = "x") # initialize AD object called "x" with the value 5
    # y = ad.AD(3, tags = "y") # initialize AD object called "y" with the value 3
    # print(x)
    # f1 = x*y              # build a function with AD objects, the function will also be an AD object
    # print(f1)             # print AD(value: {15}, derivatives: {'x': 3, 'y': 5})
    #
    # dfdx = f1.diff("x") # returns the derivative with respect to x
    # print(dfdx)                  # print 3
    #
    # jacobian1 = ad.jacobian(f1) # returns a gradient vector of f
    # print(jacobian1)  # print [3,5]
    #
    # f2 =  x + admath.sin(y)   # build a function with AD objects, the function will also be an AD object
    # print(f2)             # print AD(value: 5.141120008059867, derivatives: {'x': 1, 'y': -0.9899924966004454})
    #
    #
    # dfdy = f2.diff("y") # returns the derivative with respect to x
    # print(dfdy)                  # print -0.9899924966004454
    #
    # jacobian2 = ad.jacobian(f2) # returns a gradient vector of f
    # print(jacobian2)  # print [1, -0.9899924966004454]
    #
    # v = np.array((ad.AD(1, tags = "v_1"), ad.AD(2, tags = "v_2")))
    # u = np.array((ad.AD(3, tags = "u_1"), ad.AD(4, tags = "u_2")))
    # f = u+v
    # print(f)
    #
    #
    #
    # g = [f1,f2]
    # print(g)
    v = ad.AD([1,2], tags = "v")
    u = ad.AD([3, 4], tags="u")
    a = np.array([np.array([1,2]), np.array([3,4])])
    print(a*v)
    f = np.dot(u,v)
    print(f)

