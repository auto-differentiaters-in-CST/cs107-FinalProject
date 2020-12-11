# Use a simple (but explicit) path modification to resolve the package properly
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# import sys
# sys.path.append('../')

import pytest
import math
import numpy as np

import autodiffcst.AD as ad
from autodiffcst.AD_vec import *
# from autodiffcst.AD import AD
# from autodiffcst.AD_vec import VAD, jacobian, hessian
import autodiffcst.admath as admath


def test_initialize_base():
    x = VAD(val = [1,2])
    print(x)
    assert np.sum(x.val == np.array([1,2])) == len(x),"Error: initialize x value."
    assert np.sum(x.der == np.array([[1,0],[0,1]])) == 4,"Error: initialize x value."
    assert np.sum(x.der2 == np.array([[[0., 0.],[0., 0.]],[[0., 0.],[0., 0.]]])) == 8,"Error: initialize x value."
    with pytest.raises(TypeError):
        y = VAD([1,2], order=1.2)
    with pytest.raises(ValueError):
        y = VAD([1, 2], order=-1)
    with pytest.raises(Exception):
        y = VAD([1, 2], order=3)



def test_initialize_advanced():
    [x,y] = VAD([1,2])
    assert np.sum(x.val == np.array([1])) == len(x),"Error: initialize x value."
    assert np.sum(x.der == np.array([1., 0.])) == 2,"Error: initialize x value."
    assert np.sum(x.der2 == np.array([[0., 0.],[0., 0.]])) == 4,"Error: initialize x value."

def test_repr():
    x = VAD([1])
    print(x)
    assert repr(x) == "VAD(value: [1], tag: [0], derivatives: [[1.]], second derivatives: [[[0.]]])", "Error: repr is not working"

def test_negative():
    [x,y,z] = VAD([1,2,3])
    x = -x 
    assert np.sum(x.der == np.array([[-1, 0, 0]])) == 3, "Error: AD first derivative of negation is wrong."
    assert np.sum(x.der2 == np.array([[0., 0., 0.], [0., 0., 0.],[0., 0., 0.]])) == 9, "Error: AD second derivative of negation is wrong"

    f = VAD([3,4,5])
    f = -f
    assert np.sum(f.der == np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])) == 9, "Error: VAD first derivative of negation is wrong."
    assert np.sum(f.der2 == np.array([[[0., 0., 0.], [0., 0., 0.],[0., 0., 0.]]])) == 27, "Error: VAD second derivative of negation is wrong"

def test_eq_VAD():
    A = VAD([1,2,3])
    B = VAD([1,2,1])
    eq1 = A == B
    eq2 = A == A
    assert eq1 == False, "Error: dunder equal for VAD is wrong."
    assert eq2 == True, "Error: dunder equal for VAD is wrong."

def test_eq_AD():
    A = VAD([1,2,3])
    B = VAD([1,2,1])
    eq1 = A[0] == B[2]
    eq2 = A[0] == A[2]
    assert eq1 == True, "Error: dunder equal for VAD is wrong."
    assert eq2 == False, "Error: dunder equal for VAD is wrong."

def test_isequal_VAD():
    A = VAD([1,2,3])
    B = VAD([1,2,1])
    eq1 = A.isequal(B)
    eq2 = A.isequal(A)
    assert np.sum(eq1 == np.array([True, True, False])) == 3, "Error: isequal for VAD is wrong."
    assert np.sum(eq2 == np.array([True, True, True])) == 3, "Error: isequal for VAD is wrong."

def test_add_AD():
    # x = ad.AD(3)
    # y = ad.AD(1)
    [x, y] = VAD([3,1])
    f = x + y
    g = x + 1
    f += y
    
    assert f.val == 5, "Error: add value for AD is wrong."
    assert np.sum(f.der == np.array([1, 2])) == 2, "Error: add der for AD is wrong."
    
    assert np.sum(f.der2 == np.array([[0, 0],[0, 0]])) == 4, "Error: add der2 for AD is wrong."
    assert f-1 == g, "Error: add comparison for AD is wrong."
    
    # assert f.val == 4, "Error: add value for AD is wrong."
    # assert np.sum(f.der == np.array([1, 1])) == 2, "Error: add der for AD is wrong."
    

def test_add_VAD():
    
    f = VAD([2,1])
    m = f + 1
    g = 1 + m
    g += f
    
    h = VAD([0,0]) + VAD([6,4])
    assert np.sum(g.val == np.array([6, 4])) == 2, "Error: add value for VAD is wrong."
    print(g)
    assert np.sum(g.der == np.array([[2, 0],[0,2]])) == 4, "Error: add der for VAD is wrong."
    
    assert np.sum(g.der2 == np.array([[[0, 0],[0, 0]],[[0, 0],[0, 0]]])) == 8, "Error: add der2 for VAD is wrong."
    
    assert g == h, "Error: add value for VAD is wrong."

def test_sub_AD():
    # x = ad.AD(3)
    # y = ad.AD(1)
    [x, y] = VAD([3,1])
    f = 0 - (x - y)
    print(f.der, f.der2)
    assert f.val == -2, "Error: truediv value for AD is wrong."
    assert np.sum(f.der == np.array([-1, 1])) == 2, "Error: truediv der for AD is wrong."
    print(f.der2)
    assert np.sum(f.der2 == np.array([[0, 0],[0, 0]])) == 4, "Error: truediv der for AD is wrong."


def test_sub_VAD():
    f = VAD([3, 1])
    g = f - 4
    h = VAD([1, 1]) - VAD([2, 4])
    assert np.sum(g.val == np.array([-1, -3])) == 2, "Error: sub value for VAD is wrong."
    assert np.sum(g.der == np.array([[1, 0], [0, 1]])) == 4, "Error: sub der for VAD is wrong."

    assert np.sum(g.der2 == np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])) == 8, "Error: sub der2 for VAD is wrong."

    assert g == h, "Error: sub value for VAD is wrong."

def test_mul_AD():
    [x, y] = VAD([3,1])
    f = x * y
    g = 4 * x
    g *= y
    h = g*y

    print(f.der2)
    assert f.val == 3, "Error: mul value for AD is wrong."
    assert np.sum(f.der == np.array([1, 3])) == 2, "Error: mul der for AD is wrong."
    assert np.sum(f.der2 == np.array([[0, 1],[1, 0]])) == 4, "Error: mul der for AD is wrong."
    
    assert g.val == 12, "Error: mul value for AD is wrong."
    assert np.sum(g.der == np.array([4, 12])) == 2, "Error: mul der for AD is wrong."
    assert np.sum(g.der2 == np.array([[0, 4], [4, 0]])) == 4, "Error: mul der for AD is wrong."
    
    assert h.val == 12, "Error: mul value for AD is wrong."
    assert np.sum(h.der == np.array([4, 24])) == 2, "Error: mul der for AD is wrong."
    assert np.sum(h.der2 == np.array([[0, 8], [8, 24]])) == 4, "Error: mul der for AD is wrong."

def test_mul_VAD():
    f = VAD([3,1])
    g = 4*f
    assert np.sum(g.val == np.array([12, 4])) == 2, "Error: mul value for VAD is wrong."
    assert np.sum(g.der == np.array([[4, 0], [0, 4]])) == 4, "Error: mul der for VAD is wrong."
    assert np.sum(g.der2 == np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])) == 8, "Error: mul der2 for VAD is wrong."

def test_div_AD():
    [x, y] = VAD([3,1])
    f = x / y
    g = 9 / x
    assert f.val == 3, "Error: div value for AD is wrong."
    assert np.sum(f.der == np.array([1, -3])) == 2, "Error: div der for AD is wrong."
    assert np.sum(f.der2 == np.array([[0, -1],[-1, 6]])) == 4, "Error: div der for AD is wrong."

    assert g.val == 3, "Error: div value for AD is wrong."
    assert np.sum(g.der == np.array([-1, 0])) == 2, "Error: div der for AD is wrong."
    assert np.sum(g.der2 == np.array([[18/27, 0], [0, 0]])) == 4, "Error: div der for AD is wrong."

def test_div_VAD():
    f = VAD([3,1])
    h = f / 3
    g = 1 / f
    assert np.sum(h.val == np.array([1, 1/3])) == 2, "Error: div value for AD is wrong."
    assert np.sum(h.der == np.array([[1/3, 0], [0, 1/3]])) == 4, "Error: div der for AD is wrong."
    assert np.sum(h.der2 == np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])) == 8, "Error: div der for AD is wrong."
    print(g)
    assert np.sum(g.val == np.array([1/3, 1])) == 2, "Error: div value for AD is wrong."
    assert np.sum(g.der == np.array([[-1/9, 0], [0, -1]])) == 4, "Error: div der for AD is wrong."
    assert np.sum(g.der2 == np.array([[[2/27, 0], [0, 0]], [[0, 0], [0, 2]]])) == 8, "Error: div der for AD is wrong."

def test_pow_AD():
    [x, y] = VAD([3,1])
    f = x ** 2
    assert f.val == 9, "Error: pow value for AD is wrong."
    assert np.sum(f.der == np.array([6, 0])) == 2, "Error: pow der for AD is wrong."
    assert np.sum(f.der2 == np.array([[2, 0],[0, 0]])) == 4, "Error: pow der for AD is wrong."
    g = 2 ** x
    print(g)
    # precision is different from np
    assert np.allclose(g.val, np.array([8.0])), "Error: pow value for AD is wrong."
    assert np.allclose(g.der,np.array([8*np.log(2), 0.0])), "Error: pow der for AD is wrong."
    assert np.allclose(g.der2, np.array([[8*np.log(2)*np.log(2), 0.0], [0., 0.]])), "Error: pow der for AD is wrong."
    print(np.array([[8*np.log(2)*np.log(2), 0], [0, 0]]))

def test_set_VAD():
    x = ad.AD(1, tag=0, size=2)
    y = ad.AD(2, tag=1, size=2)
    ADs = np.array([x, y])
    assert set_VAD(ADs) == VAD([1,2]), "Error: set_VAD is wrong."

def test_decorator():
    func = my_decorator(admath.exp)
    y = VAD([1, 2, 3])
    assert func(y) == exp(y)

def test_jacobian():
    x = VAD([3, 1])
    f = 2 * x
    g = 2 * x[0]
    assert np.allclose(jacobian(f),np.array([[2., 0.],[0.,2.]]))
    assert np.allclose(jacobian(g),np.array([2., 0.]))

def test_hessian():
    x = VAD([3, 1])
    g = 2 * x[0]
    assert np.allclose(hessian(g), np.array([[0., 0.], [0., 0.]]))

def test_diff():
    x = VAD([3, 1])
    f = 2 * x
    g = x[1] * x[0]
    assert np.allclose(f.diff(1, 1), np.array([0.,2.]))
    assert np.allclose(f.diff([1, 0], 2), np.array([0.,0.]))
    assert g.diff(0, 1) == 1.0
    assert g.diff(1, 1) == 3.0
    assert g.diff([0, 0], 2) ==0.0
    with pytest.raises(Exception):
        g.diff(0, 2)


def test_fullequal():
    a = VAD([1, 2, 3])
    b = VAD([1, 2, 3]) * 2
    assert np.sum(a.fullequal(b)) ==0
    assert a[0].fullequal(a[0])
    with pytest.raises(TypeError):
        a.fullequal(1)

def test_gt():
    a = VAD([1, 2, 3])
    b = VAD([2, 2, 3])
    assert not a > b
    assert b[0] > a[0]
    with pytest.raises(TypeError):
        assert a > 1

def test_ge():
    a = VAD([1, 2, 3])
    b = VAD([2, 2, 3])
    assert not a >= b
    assert b[0] >= a[0]
    with pytest.raises(TypeError):
        assert a>= 1


def test_isgreater():
    a = VAD([1, 3, 3])
    b = VAD([2, 2, 3])
    assert np.sum(a.isgreater(b) == np.array([False, True, False])) == 3
    with pytest.raises(TypeError):
        a.isgreater(2)

def test_le():
    a = VAD([1, 1, 1])
    b = VAD([2, 2, 3])
    assert a <= b
    assert not b[0] <= a[0]
    with pytest.raises(TypeError):
        assert a <= 1

def test_lt():
    a = VAD([1, 2, 3])
    b = VAD([2, 2, 3])
    assert not a < b
    assert not b[0] <= a[0]
    with pytest.raises(TypeError):
        assert a < 1

def test_isless():
    a = VAD([1, 2, 3])
    b = VAD([2, 2, 3])
    assert np.sum(a.isless(b) == np.array([True, False, False])) ==3
    with pytest.raises(TypeError):
        a.isless(3)

def test_rsub():
    a = VAD([1])
    assert (1-a).val == 0.0


def test_mod():
    a = VAD([10])
    assert a%3 == 1
    with pytest.raises(TypeError):
        assert a%[1,2] ==1

















# def test_div_AD():
#     # x = ad.AD(3)
#     # y = ad.AD(1)
#     [x, y] = VAD([3,1])
#     f = x/y
#     print(f.der, f.der2)
#     assert f.val == 3, "Error: truediv value for AD is wrong."
#     assert np.sum(f.der == np.array([1, -3])) == 2, "Error: truediv der for AD is wrong."
#     print(f.der2)
#     assert f.der2 == np.array([[0, -1],[-1, 6]]), "Error: truediv der for AD is wrong."

# def test_add_constant():
#     A = VAD([1,2,3])
#     f = A + 3 
#     g = 3 + A
#     assert f == , "Error: AD first derivative of negation is wrong."
#     assert np.sum(x.der2 == np.array([[0., 0., 0.], [0., 0., 0.],[0., 0., 0.]])) == 9, "Error: AD second derivative of negation is wrong"

#     f = vad.VAD([3,4,5])
#     f = -f
#     assert np.sum(f.der == np.array([[-1, 0, 0], [0, -1, 0], [0, 0, -1]])) == 9, "Error: VAD first derivative of negation is wrong."
#     assert np.sum(f.der2 == np.array([[[0., 0., 0.], [0., 0., 0.],[0., 0., 0.]]])) == 27, "Error: VAD second derivative of negation is wrong"

    
    
# def test_add_constant():
#     x = AD.AD(2, "x")
#     f1 = x + 1 
#     f2 = 1 + f1
#     f3 = 1 + f1
#     f3 += 1
#     assert f1.diff("x") == 1, "Error: x+a, false derivative."
#     assert f1.val == 3, "Error: x+a, false value {f1}."

#     assert f2.diff("x") == 1, "Error: a+x, false derivative."
#     assert f2.val == 4, "Error: a+x, false value."

#     assert f3.diff("x") == 1, "Error: x+=a false derivative."
#     assert f3.val == 5, "Error: f+=a, false value."

# def test_add_variable():
#     x = AD.AD(2, "x")
#     y = AD.AD(3, "y")
#     f1 = x + y 
#     f2 = y + x
#     f3 = f1
#     f3 += x
#     assert f1.diff() == {'x': 1, 'y': 1}, "Error: x+y, false derivative."
#     assert f1.val == 5, "Error: x+y, false value."

#     assert f2.diff() == {'x': 1, 'y': 1}, "Error: a+x, false derivative."
#     assert f2.val == 5, "Error: y+x, false value."

#     assert f3.diff() == {'x': 2, 'y': 1}, "Error: x+=a false derivative."
#     assert f3.val == 7, "Error: f+=x, false value."

# def test_sub_constant():
#     x = AD.AD(2, "x")
#     f1 = x - 1 
#     f2 = 1 - x
#     f3 = f1
#     f3 -= 1
#     assert f1.diff() == {'x': 1}, "Error: x-a, false derivative."
#     assert f1.val == 1, "Error: x-a, false value."

#     assert f2.diff() == {'x': -1}, "Error: a-x, false derivative."
#     assert f2.val == -1, "Error: a-x, false value."

#     assert f3.diff() == {'x': 1}, "Error: x+=a false derivative."
#     assert f3.val == 0, "Error: f-=a, false value."

# def test_sub_variable():
#     x = AD.AD(2, "x")
#     y = AD.AD(3, "y")
#     f1 = x - y 
#     f2 = y - x
#     f3 = f1
#     f3 -= x
#     assert f1.diff() == {'x': 1, 'y': -1}, "Error: x-y, false derivative."
#     assert f1.val == -1, "Error: x-y, false value."

#     assert f2.diff() == {'x': -1, 'y': 1}, "Error: a+x, false derivative."
#     assert f2.val == 1, "Error: y-x, false value."

#     assert f3.diff() == {'x': 0, 'y': -1}, "Error: x+=a false derivative."
#     assert f3.val == -3, "Error: f-=x, false value."

# def test_mod():
#     x = AD.AD(11, "x")
#     f = x % 5

#     assert f.diff() == {'x': 1}, "Error: x mod a, false derivative."
#     assert f.val == 1, "Error: x mod a, false value."

# def test_mul_constant():
#     x = AD.AD(2, "x")
#     f1 = x * 5 
#     f2 = 2 * f1 
#     f3 = f1 * 2
#     f3 *= 1
#     assert f1.diff("x") == 5, "Error: x*a, false derivative."
#     assert f1.val == 10, "Error: x*a, false value {f1}."

#     assert f2.diff("x") == 10, "Error: a*x, false derivative."
#     assert f2.val == 20, "Error: a*x, false value."

#     assert f3.diff("x") == 10, "Error: x*=a false derivative."
#     assert f3.val == 20, "Error: f*=a, false value."

# def test_mul_variable():
#     x = AD.AD(2, "x")
#     y = AD.AD(3, "y")
#     f1 = x * y 
#     f2 = y * x
#     f3 = f1
#     f3 *= y
#     assert f1.diff() == {'x': 3, 'y': 2}, "Error: x*y, false derivative."
#     assert f1.val == 6, "Error: x*y, false value."

#     assert f2.diff() == {'x': 3, 'y': 2}, "Error: y*x, false derivative."
#     assert f2.val == 6, "Error: y*x, false value."

#     assert f3.diff() == {'x': 9, 'y': 12}, "Error: x*=a false derivative."
#     assert f3.val == 18, "Error: f*=x, false value."

# def test_div_constant():
#     x = AD.AD(6, "x")
#     f1 = x / 2
#     f2 = 2 / x
#     f3 = f1
#     f3 /= 3
#     assert f1.diff("x") == 1/2, "Error: x/a, false derivative."
#     assert f1.val == 3, "Error: x/a, false value {f1}."

#     assert f2.diff("x") == -1/18, "Error: a/x, false derivative."
#     assert f2.val == 1/3, "Error: a/x, false value."

#     assert f3.diff("x") == 1/6, "Error: x/=a false derivative."
#     assert f3.val == 1, "Error: f/=a, false value."

# def test_div_variable():
#     x = AD.AD(6, "x")
#     y = AD.AD(3, "y")
#     f1 = x / y 
#     f2 = y / x
#     f3 = f1
#     f3 /= x
#     assert f1.diff() == {'x': 1/3, 'y': -2/3}, "Error: x/y, false derivative."
#     assert f1.val == 2, "Error: x/y, false value."

#     assert f2.diff() == {'x': -1/12, 'y': 1/6}, "Error: y/x, false derivative."
#     assert f2.val == 1/2, "Error: y/x, false value."

#     assert f3.diff() == {'x': 0, 'y': -1/9}, "Error: x/=y false derivative."
#     assert f3.val == 1/3, "Error: x/=y, false value."

# def test_pow_constant():
#     x = AD.AD(2, "x")
#     f1 = x ** 2 
#     f2 = 2 ** f1
#     f3 = f1
#     f3 **= 2
#     assert f1.diff("x") == 4, "Error: x**a, false derivative."
#     assert f1.val == 4, "Error: x**a, false value {f1}."

#     assert f2.diff("x") == math.log(2) * (2**4) * 4, "Error: a**x, false derivative."
#     assert f2.val == 16, "Error: a**x, false value."

#     assert f3.diff("x") == 4*(2**3), "Error: x**=a false derivative."
#     assert f3.val == 16, "Error: f**=a, false value."

# def test_pow_variable():
#     x = AD.AD(2, "x")
#     y = AD.AD(3, "y")
#     f1 = x ** y 
#     f2 = y ** x
#     f3 = f1
#     f3 **= x
#     assert f1.diff() == {'x': 12, 'y': math.log(2) * (2**3)}, "Error: x**y, false derivative."
#     assert f1.val == 8, "Error: x**y, false value."

#     assert f2.diff() == {'x':  math.log(3) * (3**2), 'y': 6}, "Error: y**x, false derivative." 
#     assert f2.val == 9, "Error: y**x, false value." 

#     assert f3.diff() == {'x': (math.log(2)+1) * (2**6) * 3, 'y': math.log(2) * (2**6) * 2}, "Error: x**=y false derivative." 
#     assert f3.val == 64, "Error: f**=x, false value." 

# def test_jacobian():
#     x = AD.AD(2, "x")
#     y = AD.AD(3, "y")
    
#     f = x/x  
#     assert AD.jacobian(f) == [0], "Error: x/x, false jacobian."
    
#     f = x**y
#     f += y 
#     assert AD.jacobian(f) == [12, math.log(2) * (2**3)+1], "Error: x**y + y, false jacobian."
    
#     f = x/y
#     assert AD.jacobian(f) == [1/3, -2/3**2], "Error: x/y, false jacobian."
    

#     f = (math.e ** y)/x
#     assert AD.jacobian(f) == [-math.e**3 / 4, math.e**3 /2], "Error: e**y /x, false jacobian."

# def test_chain_rule():
#     ad = AD.AD(2, "x")
#     der = 2
#     new_val = 3
#     assert chain_rule(ad, new_val, der).__eq__(AD.AD(3, "x", 2))

# def test_abs():
#     x = AD.AD(1, "x")
#     f = abs(x)
#     assert f.val == 1
#     assert f.ders == {'x': 1}
#     x = AD.AD(-1, "x")
#     f = abs(x)
#     assert f.val == 1
#     assert f.ders == {'x': -1}
#     x = AD.AD(0, "x")
#     with pytest.raises(Exception):
#         f = abs(x)



# def test_log():
#     x = AD.AD(1, "x")
#     g = log(x)
#     assert g.ders == {'x': 1}
#     assert g.val == math.log(1)
#     assert g.tags == ['x']


# # trig test:
# def test_sin_float():
#     ad = 2
#     assert sin(ad).__eq__(math.sin(2)), "Error: sin(x), false value."

# def test_sin_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert sin(x+y).val == math.sin(3), "Error: sin(x+y), false value."
#     assert sin(x+y).diff() == {'x': math.cos(3), 'y': math.cos(3)}, "Error: sin(x+y), false derivative."

# def test_cos_float():
#     ad = 2
#     assert cos(ad).__eq__(math.cos(2)), "Error: cos(x), false value."

# def test_cos_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert cos(x+y).val == math.cos(3), "Error: cos(x+y), false value."
#     assert cos(x+y).diff() == {'x': -math.sin(3), 'y': -math.sin(3)}, "Error: cos(x+y), false derivative."

# def test_tan_float():
#     ad = 2
#     assert tan(ad).__eq__(math.tan(2)), "Error: tan(x), false value."

# def test_tan_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert tan(x+y).val == math.tan(3), "Error: tan(x+y), false value."
#     assert tan(x+y).diff() == {'x': 1/(math.cos(3)**2), 'y': 1/(math.cos(3)**2)}, "Error: tan(x+y), false derivative."

# def test_cot_float():
#     ad = 2
#     assert cot(ad).__eq__(1/math.tan(2)), "Error: cot(x), false value."

# epsilon = 10**(-10)

# def test_cot_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert cot(x+y).val == 1/math.tan(3), "Error: cot(x+y), false value."
#     assert abs(cot(x+y).diff('x') - (-1/(math.sin(3)**2))) <= epsilon, "Error: cot(x+y), false derivative."


# def test_sec_float():
#     ad = 2
#     assert abs(sec(ad)-1/math.cos(2)) <= epsilon, "Error: sec(x), false value."

# def test_sec_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert sec(x+y).val == 1/math.cos(3), "Error: sec(x+y), false value."
#     assert sec(x+y).diff('x') - (math.tan(3)/math.cos(3)) <= epsilon, "Error: sec(x+y), false derivative."

# def test_csc_float():
#     ad = 2
#     assert abs(csc(ad)-1/math.sin(2)) <= epsilon, "Error: csc(x), false value."

# def test_sec_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert abs(csc(x+y).val - 1/math.sin(3)) <= epsilon, "Error: csc(x+y), false value."
#     assert abs(csc(x+y).diff('x') - (-1/(math.tan(3)*math.sin(3)))) <= epsilon, "Error: csc(x+y), false derivative."

# # hyperbolic trig
# def test_sinh_float():
#     ad = 2
#     assert abs(sinh(ad)-math.sinh(2)) <= epsilon, "Error: sinh(x), false value."

# def test_sinh_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert abs(sinh(x+y).val - math.sinh(3)) <= epsilon, "Error: sinh(x+y), false value."
#     assert abs(sinh(x+y).diff('x') - math.cosh(3)) <= epsilon, "Error: sinh(x+y), false derivative."

# def test_cosh_float():
#     ad = 2
#     assert abs(cosh(ad)-math.cosh(2)) <= epsilon, "Error: cosh(x), false value."

# def test_cosh_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert abs(cosh(x+y).val - math.cosh(3)) <= epsilon, "Error: cosh(x+y), false value."
#     assert abs(cosh(x+y).diff('x') - math.sinh(3)) <= epsilon, "Error: sinh(x+y), false derivative."

# def test_tanh_float():
#     ad = 2
#     assert abs(tanh(ad)-math.tanh(2)) <= epsilon, "Error: tanh(x), false value."

# def test_tanh_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert abs(tanh(x+y).val - math.tanh(3)) <= epsilon, "Error: tanh(x+y), false value."
#     assert abs(tanh(x+y).diff('x') - 1/(math.cosh(3)**2)) <= epsilon, "Error: tanh(x+y), false derivative."

# def test_coth_float():
#     ad = 2
#     assert abs(coth(ad)-math.cosh(2)/math.sinh(2)) <= epsilon, "Error: coth(x), false value."

# def test_coth_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert abs(coth(x+y).val - math.cosh(3)/math.sinh(3)) <= epsilon, "Error: coth(x+y), false value."
#     assert abs(coth(x+y).diff('x') - (-1/(math.sinh(3)**2))) <= epsilon, "Error: coth(x+y), false derivative."

# def test_sech_float():
#     ad = 2
#     assert abs(sech(ad)-1/math.cosh(2)) <= epsilon, "Error: sech(x), false value."

# def test_sech_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert abs(sech(x+y).val - 1/math.cosh(3)) <= epsilon, "Error: sech(x+y), false value."
#     assert abs(sech(x+y).diff('x') - (-math.tanh(3)/math.cosh(3))) <= epsilon, "Error: sech(x+y), false derivative."

# def test_csch_float():
#     ad = 2
#     assert abs(csch(ad)-1/math.sinh(2)) <= epsilon, "Error: csch(x), false value."

# def test_csch_ad():
#     x = AD.AD(1, "x")
#     y = AD.AD(2, "y")
#     assert abs(csch(x+y).val - 1/math.sinh(3)) <= epsilon, "Error: csch(x+y), false value."
#     assert abs(csch(x+y).diff('x') - 1/(-math.sinh(3)*math.tanh(3))) <= epsilon, "Error: csch(x+y), false derivative."


# # trig inverse
# def test_asin_float():
#     ad = 1/2
#     assert abs(asin(ad)-math.asin(1/2)) <= epsilon, "Error: asin(x), false value."

# def test_asin_ad():
#     x = AD.AD(1/2, "x")
#     y = AD.AD(1/3, "y")
#     assert abs(asin(x+y).val - math.asin(5/6)) <= epsilon, "Error: asin(x+y), false value."
#     assert abs(asin(x+y).diff('x') - 1/math.sqrt(1-25/36)) <= epsilon, "Error: asin(x+y), false derivative."

# def test_acos_float():
#     ad = 1/2
#     assert abs(acos(ad)-math.acos(1/2)) <= epsilon, "Error: acos(x), false value."

# def test_asin_ad():
#     x = AD.AD(1/2, "x")
#     y = AD.AD(1/3, "y")
#     assert abs(acos(x+y).val - math.acos(5/6)) <= epsilon, "Error: acos(x+y), false value."
#     assert abs(acos(x+y).diff('x') + 1/math.sqrt(1-25/36)) <= epsilon, "Error: acos(x+y), false derivative."

# def test_atan_float():
#     ad = 1/2
#     assert abs(atan(ad)-math.atan(1/2)) <= epsilon, "Error: atan(x), false value."

# def test_atan_ad():
#     x = AD.AD(1/2, "x")
#     y = AD.AD(1/3, "y")
#     assert abs(atan(x+y).val - math.atan(5/6)) <= epsilon, "Error: atan(x+y), false value."
#     assert abs(atan(x+y).diff('x') - 1/(1+25/36)) <= epsilon, "Error: atan(x+y), false derivative."

# def test_acot_float():
#     ad = 1/2
#     assert abs(acot(ad)-math.atan(2)) <= epsilon, "Error: acot(x), false value."

# def test_acot_ad():
#     x = AD.AD(1/2, "x")
#     y = AD.AD(1/3, "y")
#     assert abs(acot(x*y).val - math.atan(6)) <= epsilon, "Error: acot(x*y), false value."
#     assert abs(acot(x*y).diff('x') + (1/3) * 1/(1+1/36)) <= epsilon, "Error: acot(x*y), false derivative."

# def test_asec_float():
#     ad = 2
#     assert abs(asec(ad)-math.acos(1/2)) <= epsilon, "Error: asec(x), false value."

# def test_asec_ad():
#     x = AD.AD(2, "x")
#     y = AD.AD(3, "y")
#     assert abs(asec(x*y).val - math.acos(1/6)) <= epsilon, "Error: asec(x*y), false value."
#     assert abs(asec(x*y).diff('x') - 3/(6*math.sqrt(36-1))) <= epsilon, "Error: asec(x*y), false derivative."

# def test_acsc_float():
#     ad = 2
#     assert abs(acsc(ad)-math.asin(1/2)) <= epsilon, "Error: acsc(x), false value."

# def test_acsc_ad():
#     x = AD.AD(2, "x")
#     y = AD.AD(3, "y")
#     assert abs(acsc(x*y).val - math.asin(1/6)) <= epsilon, "Error: acsc(x*y), false value."
#     assert abs(acsc(x*y).diff('x') + 3/(6*math.sqrt(36-1))) <= epsilon, "Error: acsc(x*y), false derivative."


# # hyperbolic trig inverse
# def test_asinh_float():
#     ad = 1/2
#     assert abs(asinh(ad)-math.asinh(1/2)) <= epsilon, "Error: asinh(x), false value."

# def test_asinh_ad():
#     x = AD.AD(1/2, "x")
#     y = AD.AD(1/3, "y")
#     assert abs(asinh(x+y).val - math.asinh(5/6)) <= epsilon, "Error: asinh(x+y), false value."
#     assert abs(asinh(x+y).diff('x') - 1/math.sqrt(25/36+1)) <= epsilon, "Error: asinh(x+y), false derivative."

# def test_acosh_float():
#     ad = 2
#     assert abs(acosh(ad)-math.acosh(2)) <= epsilon, "Error: acosh(x), false value."

# def test_acosh_ad():
#     x = AD.AD(2, "x")
#     y = AD.AD(3, "y")
#     assert abs(acosh(x+y).val - math.acosh(5)) <= epsilon, "Error: acosh(x+y), false value."
#     assert abs(acosh(x+y).diff('x') - 1/math.sqrt(25-1)) <= epsilon, "Error: acosh(x+y), false derivative."

# def test_atanh_float():
#     ad = 1/2
#     assert abs(atanh(ad)-math.atanh(1/2)) <= epsilon, "Error: atanh(x), false value."

# def test_atanh_ad():
#     x = AD.AD(1/2, "x")
#     y = AD.AD(1/3, "y")
#     assert abs(atanh(x+y).val - math.atanh(5/6)) <= epsilon, "Error: atanh(x+y), false value."
#     assert abs(atanh(x+y).diff('x') - 1/(1-25/36)) <= epsilon, "Error: atanh(x+y), false derivative."

# def test_acoth_float():
#     ad = 2
#     assert abs(acoth(ad)-0.5*math.log((ad+1)/(ad-1))) <= epsilon, "Error: acoth(x), false value."

# def test_acoth_ad():
#     x = AD.AD(2, "x")
#     y = AD.AD(3, "y")
#     assert abs(acoth(x*y).val - 0.5*math.log((6+1)/(6-1))) <= epsilon, "Error: acoth(x*y), false value."
#     assert abs(acoth(x*y).diff('x') - 3 * 1/(1-36)) <= epsilon, "Error: acoth(x*y), false derivative."

# def test_asech_float():
#     ad = 1/2
#     assert abs(asech(ad)-math.log((1+math.sqrt(1-ad**2))/ad)) <= epsilon, "Error: asech(x), false value."

# def test_asech_ad():
#     x = AD.AD(1/2, "x")
#     y = AD.AD(1/3, "y")
#     assert abs(asech(x*y).val - math.log((1+math.sqrt(1-1/36))/(1/6))) <= epsilon, "Error: asech(x*y), false value."
#     assert abs(asech(x*y).diff('x') + (1/3)/((1/6)*math.sqrt(1-1/36))) <= epsilon, "Error: asech(x*y), false derivative."

# def test_acsch_float():
#     ad = 2
#     assert abs(acsch(ad)-math.log(1/2+math.sqrt(1/4+1))) <= epsilon, "Error: acsch(x), false value."

# def test_acsch_ad():
#     x = AD.AD(2, "x")
#     y = AD.AD(3, "y")
#     assert abs(acsch(x*y).val - math.log(1/6+math.sqrt(1/36+1))) <= epsilon, "Error: acsch(x*y), false value."
#     assert abs(acsch(x*y).diff('x') + 3/(6*math.sqrt(1+36))) <= epsilon, "Error: acsch(x*y), false derivative."
