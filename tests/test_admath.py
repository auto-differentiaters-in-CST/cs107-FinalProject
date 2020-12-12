# Use a simple (but explicit) path modification to resolve the package properly
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '../src')))

import pytest
import math
import numpy as np
# import src.autodiffcst.AD as AD
# from src.autodiffcst.trigmath import *
import autodiffcst.AD as AD
from autodiffcst.AD_vec import *
import autodiffcst.admath as admath

# def test_set_VAD():
#     vadtest = VAD([1,2,3])
#     ad1 = vadtest[0]
#     ad2 = vadtest[1]
#     ad3 = vadtest[2]
#     new_vad = set_VAD(np.array([ad1,ad2,ad3]))
#     assert new_vad == vadtest, "Error: returned object not good."
#     assert sin(vadtest) == set_VAD(np.array([sin(ad1),sin(ad2),sin(ad3)])),"Error: returned object not good."

def test_abs():
    x = VAD([-1,2,3])
    f = abs(x)
    print(f.der)
    print(f.der2)
    assert np.sum(f.val == np.array([[1],[2],[3]])) == 3, "Error: abs didn't apply properly on VAD."
    assert np.sum(f.der == np.array([[-1., 0., 0.],[0., 1., 0.],[0., 0., 1.]])) == 9, "Error: der1 for abs(VAD) is not correct."
    x = AD.AD(-1, tag=0)
    f = abs(x)
    assert f.val == 1, "Error: abs didn't apply properly on AD."
    assert f.der == -1, "Error: der1 for abs(AD) is not correct."
    assert abs(-5) == 5, "Error: abs didn't apply properly on numbers."
    x = AD.AD(0, tag=0)
    with pytest.raises(Exception):
        f = abs(x)
    with pytest.raises(TypeError):
        f = admath.abs('error')
    xv = AD.AD(-2,tag=0,order=3)
    f = admath.abs(xv)
    assert f.higherdiff(3) == 0, "Error: abs didn't apply on higher properly"
    [x] = VAD([-5],order=3)
    f = abs(x)
    print(f.der)
    print(f.der2)
    print(f.higher)

def test_chain_rule():
    x = AD.AD(2,order=5)
    newad = 5*x**3+2
    higherde = np.array([60,60,30,0,0])
    adres = admath.chain_rule(x, 42, 60, 60, higher_der=higherde)
    assert newad == adres, "Error: chain rule doesn't apply to AD object properly."
    assert np.allclose(newad.higher,adres.higher), "Error: chain rule doesn't carry higher derivatives properly."
    # x = AD.AD(2,order=5)
    # newad = 5*x**3+2
    # higherde = np.array([60,60,30,0,0])
    # adres = chain_rule(x, 40, 60, 60, higher_der=higherde)
    # assert newad == adres, "Error: chain rule doesn't apply to AD object properly."
    # assert newad.higher == adres.higher, "Error: chain rule doesn't carry higher derivatives properly."

def test_choose():
    assert admath.choose(6,0) == 1, "Error: choose function doesn't calculate correctly."
    assert admath.choose(10,1) == 10, "Error: choose function doesn't calculate correctly."
    assert admath.choose(9,9) == 1, "Error: choose function doesn't calculate correctly."
    assert admath.choose(3,2) == 3, "Error: choose function doesn't calculate correctly."
    with pytest.raises(Exception):
        admath.choose(1,-1)
    with pytest.raises(Exception):
        admath.choose(5,11)
    with pytest.raises(Exception):
        admath.choose(4.5,1.1)


def test_log():
    x = AD.AD(1, tag=0,order=5)
    f = log(x)
    assert f.val[0] == np.log(1), "Error: log didn't apply properly on AD."
    assert np.allclose(f.der,1), "Error: der1 for log(AD) is not correct."
    assert np.allclose(f.der2,-1), "Error: der2 for log(AD) is not correct."
    higherarr = np.array([1,-1,2,-6,24])
    assert np.allclose(f.higher,higherarr), "Error: higherder for log(AD) is not correct."
    y = VAD([1,2,3])
    g = log(y)
    assert np.allclose(g.val, np.log([1,2,3])), "Error: log didn't apply properly on VAD."
    dertest = np.zeros((3,3))
    dertest[0,0] = 1
    dertest[1,1] = 1/2
    dertest[2,2] = 1/3
    assert np.allclose(g.der,dertest), "Error: der1 for log(VAD) is not correct."
    der2test = np.zeros((3,3,3))
    der2test[0,0,0] = -1**-2
    der2test[1,1,1] = -2**-2
    der2test[2,2,2] = -3**-2
    assert np.allclose(g.der2,der2test), "Error: der2 for log(VAD) is not correct."
    assert log(5) == np.log(5), "Error: log didn't apply properly on number."
    with pytest.raises(TypeError):
        f = admath.log('error')

def test_fact_ad():
    assert admath.fact_ad(2,0) == 1, "Error: fact_ad calculation wrong."
    assert admath.fact_ad(2,2) == 2, "Error: fact_ad calculation wrong."
    assert admath.fact_ad(2,3) == 0, "Error: fact_ad calculation wrong."
    assert admath.fact_ad(3.5,2) == 3.5*2.5, "Error: fact_ad calculation wrong."

def test_exp():
    x = AD.AD(1, tag=0,order=5)
    f = exp(x)
    assert f.val[0] == np.exp(1), "Error: exp didn't apply properly on AD."
    assert np.allclose(f.der,np.exp(1)), "Error: der1 for exp(AD) is not correct."
    assert np.allclose(f.der2,np.exp(1)), "Error: der2 for exp(AD) is not correct."
    higherarr = np.array([np.exp(1),np.exp(1),np.exp(1),np.exp(1),np.exp(1)])
    assert np.allclose(f.higher,higherarr), "Error: higherder for exp(AD) is not correct."
    y = VAD([1,2,3])
    g = exp(y)
    assert np.allclose(g.val, np.exp([1,2,3])), "Error: exp didn't apply properly on VAD."
    dertest = np.zeros((3,3))
    dertest[0,0] = np.exp(1)
    dertest[1,1] = np.exp(2)
    dertest[2,2] = np.exp(3)
    assert np.allclose(g.der,dertest), "Error: der1 for exp(VAD) is not correct."
    der2test = np.zeros((3,3,3))
    der2test[0,0,0] = np.exp(1)
    der2test[1,1,1] = np.exp(2)
    der2test[2,2,2] = np.exp(3)
    assert np.allclose(g.der2,der2test), "Error: der2 for exp(VAD) is not correct."
    assert exp(5) == np.exp(5), "Error: exp didn't apply properly on number."
    with pytest.raises(TypeError):
        f = admath.exp('error')

def test_sqrt():
    x = AD.AD(2, tag=0,order=3)
    f = sqrt(x)
    assert f.val[0] == np.sqrt(2), "Error: sqrt didn't apply properly on AD."
    assert np.allclose(f.der,0.5*1/np.sqrt(2)), "Error: der1 for sqrt(AD) is not correct."
    assert np.allclose(f.der2,-0.25/2/np.sqrt(2)), "Error: der2 for sqrt(AD) is not correct."
    assert np.abs(f.higherdiff(3) - 3/32/np.sqrt(2))<1e-8, "Error: higherder for sqrt(AD) is not correct."
    y = VAD([1,2])
    g = sqrt(y)
    assert np.allclose(g.val, np.sqrt([1,2])), "Error: sqrt didn't apply properly on VAD."
    dertest = np.zeros((2,2))
    dertest[0,0] = 0.5
    dertest[1,1] = 0.5*1/np.sqrt(2)
    assert np.allclose(g.der,dertest), "Error: der1 for sqrt(VAD) is not correct."
    der2test = np.zeros((2,2,2))
    der2test[0,0,0] = -0.25
    der2test[1,1,1] = -0.25/2/np.sqrt(2)
    assert np.allclose(g.der2,der2test), "Error: der2 for sqrt(VAD) is not correct."
    assert sqrt(5) == np.sqrt(5), "Error: sqrt didn't apply properly on number."
    with pytest.raises(TypeError):
        f = admath.sqrt('error')

# trig test:

def test_sin_float():
    ad = 2
    assert sin(ad).__eq__(np.sin(2)), "Error: sin(x), false value."
    with pytest.raises(TypeError):
        f = admath.sin('error')

def test_sin_ad():
    [x,y] = VAD([1,2])
    assert sin(x+y).val[0] == np.sin(3), "Error: sin(x+y), false value."
    assert np.allclose(jacobian(sin(x+y)),np.array([np.cos(3),np.cos(3)])), "Error: sin(x+y), false derivative."
    f = sin(VAD([1,2]))
    assert np.allclose(f.val,np.array([np.sin(1),np.sin(2)])), "Error: sin(VAD[1,2]), false value."
    assert np.allclose(sin(x+y).der2,np.array([[-np.sin(3),-np.sin(3)],[-np.sin(3),-np.sin(3)]])), "Error: sin(x+y), false der2."
    der1 = np.zeros((2,2))
    der1[0,0] = np.cos(1)
    der1[1,1] = np.cos(2)
    der2 = np.zeros((2,2,2))
    der2[0,0,0] = -np.sin(1)
    der2[1,1,1] = -np.sin(2)
    assert np.allclose(f.der,der1), "Error: sin(VAD[1,2]), false der1."
    assert np.allclose(f.der2,der2), "Error: sin(VAD[1,2]), false der2."
    [z] = VAD([np.pi/6],order=5)
    cosv = np.sqrt(3)/2
    sinv = 1/2
    higherde = np.array([cosv,-sinv,-cosv,sinv,cosv])
    h = sin(z)
    assert np.allclose(h.higher,higherde), "Error: sin(pi/6), false higher."

def test_cos_float():
    ad = 2
    assert cos(ad).__eq__(np.cos(2)), "Error: cos(x), false value."
    with pytest.raises(TypeError):
        f = admath.cos('error')

def test_cos_ad():
    [x,y] = VAD([1,2])
    assert cos(x+y).val[0] == np.cos(3), "Error: cos(x+y), false value."
    assert np.allclose(jacobian(cos(x+y)),np.array([-np.sin(3),-np.sin(3)])), "Error: cos(x+y), false derivative."
    f = cos(VAD([1,2]))
    assert np.allclose(cos(x+y).der2,np.array([[-np.cos(3),-np.cos(3)],[-np.cos(3),-np.cos(3)]])), "Error: cos(x+y), false der2."
    assert np.allclose(f.val,np.array([np.cos(1),np.cos(2)])), "Error: cos(VAD[1,2]), false value."
    der1 = np.zeros((2,2))
    der1[0,0] = -np.sin(1)
    der1[1,1] = -np.sin(2)
    der2 = np.zeros((2,2,2))
    der2[0,0,0] = -np.cos(1)
    der2[1,1,1] = -np.cos(2)
    assert np.allclose(f.der,der1), "Error: cos(VAD[1,2]), false der1."
    assert np.allclose(f.der2,der2), "Error: cos(VAD[1,2]), false der2."
    [z] = VAD([np.pi/6],order=5)
    cosv = np.sqrt(3)/2
    sinv = 1/2
    higherde = np.array([-sinv,-cosv,sinv,cosv,-sinv])
    h = cos(z)
    assert np.allclose(h.higher,higherde), "Error: cos(pi/6), false higher."

def test_tan_float():
    ad = 2
    assert tan(ad).__eq__(np.tan(2)), "Error: tan(x), false value."
    with pytest.raises(TypeError):
        f = admath.tan('error')

def test_tan_ad():
    [x,y] = VAD([1,2])
    assert tan(x+y).val[0] == np.tan(3), "Error: tan(x+y), false value."
    assert np.allclose(jacobian(tan(x+y)),np.array([admath.sec(3)**2,admath.sec(3)**2])), "Error: tan(x+y), false derivative."
    f = tan(VAD([1,2]))
    assert np.allclose(f.val,np.array([np.tan(1),np.tan(2)])), "Error: tan(VAD[1,2]), false value."
    der1 = np.zeros((2,2))
    assert np.allclose(tan(x+y).der2,np.array([[2*tan(3)*admath.sec(3)**2,2*tan(3)*admath.sec(3)**2],[2*tan(3)*admath.sec(3)**2,2*tan(3)*admath.sec(3)**2]])), "Error: tan(x+y), false der2."
    der1[0,0] = admath.sec(1)**2
    der1[1,1] = admath.sec(2)**2
    der2 = np.zeros((2,2,2))
    der2[0,0,0] = 2*tan(1)*admath.sec(1)**2
    der2[1,1,1] = 2*tan(2)*admath.sec(2)**2
    assert np.allclose(f.der,der1), "Error: tan(VAD[1,2]), false der1."
    assert np.allclose(f.der2,der2), "Error: tan(VAD[1,2]), false der2."
    [z] = VAD([1],order=3)
    h = tan(z)
    calc = 2*admath.sec(1)**2*(2*tan(1)**2+admath.sec(1)**2)
    assert np.abs(h.higherdiff(3)-calc)<1e-8, "Error: tan(VAD[1,2]), false higher."

def test_sec_float():
    ad = 2
    assert abs(admath.sec(ad)-1/math.cos(2)) <= 1e-8, "Error: sec(x), false value."
    with pytest.raises(TypeError):
        f = admath.sec('error')

# hyperbolic trig
def test_sinh_float():
    ad = 2
    assert abs(sinh(ad)-math.sinh(2)) <= 1e-8, "Error: sinh(x), false value."
    with pytest.raises(TypeError):
        f = admath.sinh('error')

def test_sinh_ad():
    [x,y] = VAD([1,2])
    assert sinh(x+y).val[0] == np.sinh(3), "Error: sinh(x+y), false value."
    assert np.allclose(jacobian(sinh(x+y)),np.array([np.cosh(3),np.cosh(3)])), "Error: sinh(x+y), false derivative."
    f = sinh(VAD([1,2]))
    assert np.allclose(sinh(x+y).der2,np.array([[np.sinh(3),np.sinh(3)],[np.sinh(3),np.sinh(3)]])), "Error: cosh(x+y), false der2."
    assert np.allclose(f.val,np.array([np.sinh(1),np.sinh(2)])), "Error: cosh(VAD[1,2]), false value."
    der1 = np.zeros((2,2))
    der1[0,0] = np.cosh(1)
    der1[1,1] = np.cosh(2)
    der2 = np.zeros((2,2,2))
    der2[0,0,0] = np.sinh(1)
    der2[1,1,1] = np.sinh(2)
    assert np.allclose(f.der,der1), "Error: sinh(VAD[1,2]), false der1."
    assert np.allclose(f.der2,der2), "Error: sinh(VAD[1,2]), false der2."
    [z] = VAD([0],order=5)
    cosv = 1.0
    sinv = 0.0
    higherde = np.array([cosv,sinv,cosv,sinv,cosv])
    h = sinh(z)
    assert np.allclose(h.higher,higherde), "Error: sinh(0), false higher."

def test_cosh_float():
    ad = 2
    assert abs(cosh(ad)-math.cosh(2)) <= 1e-8, "Error: cosh(x), false value."
    with pytest.raises(TypeError):
        f = admath.cosh('error')

def test_cosh_ad():
    [x,y] = VAD([1,2])
    assert cosh(x+y).val[0] == np.cosh(3), "Error: cosh(x+y), false value."
    assert np.allclose(jacobian(cosh(x+y)),np.array([np.sinh(3),np.sinh(3)])), "Error: cosh(x+y), false derivative."
    f = cosh(VAD([1,2]))
    assert np.allclose(cosh(x+y).der2,np.array([[np.cosh(3),np.cosh(3)],[np.cosh(3),np.cosh(3)]])), "Error: cosh(x+y), false der2."
    assert np.allclose(f.val,np.array([np.cosh(1),np.cosh(2)])), "Error: cosh(VAD[1,2]), false value."
    der1 = np.zeros((2,2))
    der1[0,0] = np.sinh(1)
    der1[1,1] = np.sinh(2)
    der2 = np.zeros((2,2,2))
    der2[0,0,0] = np.cosh(1)
    der2[1,1,1] = np.cosh(2)
    assert np.allclose(f.der,der1), "Error: cosh(VAD[1,2]), false der1."
    assert np.allclose(f.der2,der2), "Error: cosh(VAD[1,2]), false der2."
    [z] = VAD([0],order=5)
    cosv = 1.0
    sinv = 0.0
    higherde = np.array([sinv,cosv,sinv,cosv,sinv])
    h = cosh(z)
    assert np.allclose(h.higher,higherde), "Error: cosh(0), false higher."

def test_tanh_float():
    ad = 2
    assert abs(tanh(ad)-np.tanh(2)) <= 1e-8, "Error: tanh(x), false value."
    with pytest.raises(TypeError):
        f = admath.tanh('error')

def test_tanh_ad():
    [x,y] = VAD([1,2])
    assert tanh(x+y).val[0] == np.tanh(3), "Error: tanh(x+y), false value."
    assert np.allclose(jacobian(tanh(x+y)),np.array([admath.sech(3)**2,admath.sech(3)**2])), "Error: tanh(x+y), false derivative."
    assert np.allclose(tanh(x+y).der2,np.array([[-2*tanh(3)*admath.sech(3)**2,-2*tanh(3)*admath.sech(3)**2],[-2*tanh(3)*admath.sech(3)**2,-2*tanh(3)*admath.sech(3)**2]])), "Error: tanh(x+y), false der2."
    f = tanh(VAD([1,2]))
    assert np.allclose(f.val,np.array([np.tanh(1),np.tanh(2)])), "Error: tanh(VAD[1,2]), false value."
    der1 = np.zeros((2,2))
    der1[0,0] = admath.sech(1)**2
    der1[1,1] = admath.sech(2)**2
    der2 = np.zeros((2,2,2))
    der2[0,0,0] = -2*tanh(1)*admath.sech(1)**2
    der2[1,1,1] = -2*tanh(2)*admath.sech(2)**2
    assert np.allclose(f.der,der1), "Error: tanh(VAD[1,2]), false der1."
    assert np.allclose(f.der2,der2), "Error: tanh(VAD[1,2]), false der2."
    [z] = VAD([1],order=3)
    h = tanh(z)
    calc = 4*tanh(1)**2*admath.sech(1)**2 - 2*admath.sech(1)**4
    assert np.abs(h.higherdiff(3)-calc)<1e-8, "Error: tanh(VAD[1,2]), false higher."

def test_sech_float():
    ad = 2
    assert abs(admath.sech(ad)-1/np.cosh(2)) <= 1e-8, "Error: sech(x), false value."
    with pytest.raises(TypeError):
        f = admath.sech('error')
