# Use a simple (but explicit) path modification to resolve the package properly
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))
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
    assert repr(x) == "VAD(value: [1], derivatives: [[1.]])", "Error: repr is not working"
    assert x.__repr__() == "VAD(value: [1], derivatives: [[1.]])", "Error: repr is not working"
    assert x.__str__() == "VAD(value: [1], derivatives: [[1.]])", "Error: str is not working"

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
    with pytest.raises(TypeError):
        A == 1
        A > 1
        A < 1
        A >= 1
        A <= 1
    
def test_len_VAD():
    A = VAD([1,2,3])
    l = len(A)
    assert l == 3, "Error: dunder length for VAD is wrong."
    
def test_len_AD():
    [x, y] = VAD([1,2])
    lx = len(x)
    assert lx == 1, "Error: dunder length for AD is wrong."
    

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
    assert np.sum(g.der == np.array([[2, 0],[0,2]])) == 4, "Error: add der for VAD is wrong."
    
    assert np.sum(g.der2 == np.array([[[0, 0],[0, 0]],[[0, 0],[0, 0]]])) == 8, "Error: add der2 for VAD is wrong."
    
    assert g == h, "Error: add value for VAD is wrong."

def test_sub_AD():
    # x = ad.AD(3)
    # y = ad.AD(1)
    [x, y] = VAD([3,1])
    f = 0 - (x - y)
    assert f.val == -2, "Error: truediv value for AD is wrong."
    assert np.sum(f.der == np.array([-1, 1])) == 2, "Error: truediv der for AD is wrong."
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
    g *= 2
    assert np.sum(g.val == np.array([24, 8])) == 2, "Error: mul value for VAD is wrong."
    assert np.sum(g.der == np.array([[8, 0], [0, 8]])) == 4, "Error: mul der for VAD is wrong."
    assert np.sum(g.der2 == np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])) == 8, "Error: mul der2 for VAD is wrong."

def test_div_AD():
    [x, y] = VAD([3,1])
    f = x / y
    g = 9 / x
    g /= 1
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
    g /= 1
    assert np.sum(h.val == np.array([1, 1/3])) == 2, "Error: div value for AD is wrong."
    assert np.sum(h.der == np.array([[1/3, 0], [0, 1/3]])) == 4, "Error: div der for AD is wrong."
    assert np.sum(h.der2 == np.array([[[0, 0], [0, 0]], [[0, 0], [0, 0]]])) == 8, "Error: div der for AD is wrong."
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
    # precision is different from np
    assert np.allclose(g.val, np.array([8.0])), "Error: pow value for AD is wrong."
    assert np.allclose(g.der,np.array([8*np.log(2), 0.0])), "Error: pow der for AD is wrong."
    assert np.allclose(g.der2, np.array([[8*np.log(2)*np.log(2), 0.0], [0., 0.]])), "Error: pow der for AD is wrong."

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


def test_pow_m():
    x = VAD([1, 2])
    g = pow(x, 2)
    assert np.allclose(g.val, np.array([1., 4.]))
    assert np.allclose(g.der, np.array([[2., 0.],[0., 4.]]))

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

def test_sub_m():
    a = VAD([1])
    b = 1 - a
    b -= 1
    assert b.val == -1.0
    assert b.der == -1.0


def test_mod():
    a = VAD([10])
    assert a%3 == 1
    with pytest.raises(TypeError):
        assert a%[1,2] ==1

def test_AD_init_hw():
    with pytest.raises(TypeError):
        x = ad.AD(-1,tag=0,size=1.5)
        x = ad.AD(-1,tag=0,order=1.5)
        x = ad.AD(-1,tag=0,order='error')
        x = VAD([-1],order='error')
    with pytest.raises(ValueError):
        x = ad.AD(-1,order=0)
    with pytest.raises(Exception):
        x = VAD([1,2,3],order=5)
        x = ad.AD([1,2,3],order=5)

def test_AD_repr_hw():
    x = ad.AD([1],tag=0,der=[1],order=5)
    assert x.__repr__() == 'AD(value: [[1]], derivatives: [1.])', "Error: repr is not working"
    assert x.__str__() == 'AD(value: [[1]], derivatives: [1.])', "Error: str is not working"

def test_AD_eq_hw():
    x = ad.AD([1],tag=0,der=[1],order=5)
    y = VAD([1])
    with pytest.raises(TypeError):
        x == y
        x.fullequal(y)
        x < y
        x > y
        x <= y
        x >= y
        y = VAD([1])
    
    y = 0
    with pytest.raises(TypeError):
        x == y
        x.fullequal(y)
        x < y
        x > y
        x <= y
        x >= y

def test_AD_ne_hw():
    x = ad.AD([1],tag=0,der=[1],order=5)
    assert not x != x, "Error: ne not working for AD"

def test_AD_ueq_hw():
    x = ad.AD([1],tag=0,der=[1],order=5)
    y = 2*x
    assert x != y, "Error: ne not working for AD"
    assert x < y, "Error: ne not working for AD"
    assert y > x, "Error: ne not working for AD"
    assert x <= y, "Error: ne not working for AD"
    assert y >= x, "Error: ne not working for AD"

def test_AD_add_hw():
    x = ad.AD([1],tag=0,der=[1],order=5)
    y = ad.AD([1],tag=0,der=[1],order=7)
    f = x + x
    with pytest.raises(Exception):
        f = x + y

def test_higher_hw():
    x = ad.AD(val = 1, order = 10, size = 1, tag = 0)
    f = x**5
    assert f.higherdiff(1) == 5,"higherdiff not working"
    with pytest.raises(TypeError):
        f.higherdiff('error')
    with pytest.raises(ValueError):
        f.higherdiff(0)
        f.higherdiff(15)
    with pytest.raises(Exception):
        x = ad.AD(val = 1, size = 1, tag = 0)
        f = x**5
        f.higherdiff(10)

def test_AD_rpow_hw():
    x = ad.AD(val = 1, order = 10, size = 1, tag = 0)
    f = 5**x
    assert np.abs(f.val - 5)<1e-8, "rpow not working for AD"

def test_AD_pow_hw():
    x = ad.AD(val = 1, order = 10, size = 1, tag = 0)
    f = x**5

    assert np.abs(f.val - 1)<1e-8, "rpow not working for AD"
    x = ad.AD(val = 0, order = 10, size = 1, tag = 0)
    
    with pytest.raises(ValueError):
        f = x ** 5
        f = x**2
    with pytest.raises(TypeError):
        f = x**'a'
    
def test_VAD_op_hw():
    x = VAD(val = [1,2])
    y = VAD(val = [1,2])
    z = VAD(val = [1,2])
    x -= 1
    assert np.allclose(x.val,np.array([0,1])), 'Error: wrong isub VAD'
    y %= 2
    assert np.allclose(y,np.array([1,0])), 'Error: wrong imod VAD'
    z *= 2
    assert np.allclose(z.val,np.array([2,4])), 'Error: wrong imul VAD'
    x = VAD(val = [2,2])
    x /= 2
    assert np.allclose(x.val,np.array([1,1])), 'Error: wrong itruediv VAD'
    x = VAD(val = [2,2])
    f = x**2
    assert np.allclose(f.val,np.array([4,4])), 'Error: wrong pow VAD'