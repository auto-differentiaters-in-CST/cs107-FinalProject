# Use a simple (but explicit) path modification to resolve the package properly
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import pytest
import math
# import src.autodiffcst.AD as AD
# from src.autodiffcst.trigmath import *
import autodiffcst.AD as AD
from autodiffcst.admath import *

def test_repr():
    x = AD.AD(2, "x")
    assert repr(x) == "AD(value: 2, derivatives: {'x': 1})", "Error: repr is not working"

def test_add_constant():
    x = AD.AD(2, "x")
    f1 = x + 1 
    f2 = 1 + f1
    f3 = 1 + f1
    f3 += 1
    assert f1.diff("x") == 1, "Error: x+a, false derivative."
    assert f1.val == 3, "Error: x+a, false value {f1}."

    assert f2.diff("x") == 1, "Error: a+x, false derivative."
    assert f2.val == 4, "Error: a+x, false value."

    assert f3.diff("x") == 1, "Error: x+=a false derivative."
    assert f3.val == 5, "Error: f+=a, false value."

def test_add_variable():
    x = AD.AD(2, "x")
    y = AD.AD(3, "y")
    f1 = x + y 
    f2 = y + x
    f3 = f1
    f3 += x
    assert f1.diff() == {'x': 1, 'y': 1}, "Error: x+y, false derivative."
    assert f1.val == 5, "Error: x+y, false value."

    assert f2.diff() == {'x': 1, 'y': 1}, "Error: a+x, false derivative."
    assert f2.val == 5, "Error: y+x, false value."

    assert f3.diff() == {'x': 2, 'y': 1}, "Error: x+=a false derivative."
    assert f3.val == 7, "Error: f+=x, false value."

def test_sub_constant():
    x = AD.AD(2, "x")
    f1 = x - 1 
    f2 = 1 - x
    f3 = f1
    f3 -= 1
    assert f1.diff() == {'x': 1}, "Error: x-a, false derivative."
    assert f1.val == 1, "Error: x-a, false value."

    assert f2.diff() == {'x': -1}, "Error: a-x, false derivative."
    assert f2.val == -1, "Error: a-x, false value."

    assert f3.diff() == {'x': 1}, "Error: x+=a false derivative."
    assert f3.val == 0, "Error: f-=a, false value."

def test_sub_variable():
    x = AD.AD(2, "x")
    y = AD.AD(3, "y")
    f1 = x - y 
    f2 = y - x
    f3 = f1
    f3 -= x
    assert f1.diff() == {'x': 1, 'y': -1}, "Error: x-y, false derivative."
    assert f1.val == -1, "Error: x-y, false value."

    assert f2.diff() == {'x': -1, 'y': 1}, "Error: a+x, false derivative."
    assert f2.val == 1, "Error: y-x, false value."

    assert f3.diff() == {'x': 0, 'y': -1}, "Error: x+=a false derivative."
    assert f3.val == -3, "Error: f-=x, false value."

def test_mod():
    x = AD.AD(11, "x")
    f = x % 5

    assert f.diff() == {'x': 1}, "Error: x mod a, false derivative."
    assert f.val == 1, "Error: x mod a, false value."

def test_mul_constant():
    x = AD.AD(2, "x")
    f1 = x * 5 
    f2 = 2 * f1 
    f3 = f1 * 2
    f3 *= 1
    assert f1.diff("x") == 5, "Error: x*a, false derivative."
    assert f1.val == 10, "Error: x*a, false value {f1}."

    assert f2.diff("x") == 10, "Error: a*x, false derivative."
    assert f2.val == 20, "Error: a*x, false value."

    assert f3.diff("x") == 10, "Error: x*=a false derivative."
    assert f3.val == 20, "Error: f*=a, false value."

def test_mul_variable():
    x = AD.AD(2, "x")
    y = AD.AD(3, "y")
    f1 = x * y 
    f2 = y * x
    f3 = f1
    f3 *= y
    assert f1.diff() == {'x': 3, 'y': 2}, "Error: x*y, false derivative."
    assert f1.val == 6, "Error: x*y, false value."

    assert f2.diff() == {'x': 3, 'y': 2}, "Error: y*x, false derivative."
    assert f2.val == 6, "Error: y*x, false value."

    assert f3.diff() == {'x': 9, 'y': 12}, "Error: x*=a false derivative."
    assert f3.val == 18, "Error: f*=x, false value."

def test_div_constant():
    x = AD.AD(6, "x")
    f1 = x / 0
    f2 = 2 / x
    f3 = f1
    f3 /= 3
    assert f1.diff("x") == 1/2, "Error: x/a, false derivative."
    assert f1.val == 3, "Error: x/a, false value {f1}."

    assert f2.diff("x") == -1/18, "Error: a/x, false derivative."
    assert f2.val == 1/3, "Error: a/x, false value."

    assert f3.diff("x") == 1/6, "Error: x/=a false derivative."
    assert f3.val == 1, "Error: f/=a, false value."

def test_div_variable():
    x = AD.AD(6, "x")
    y = AD.AD(3, "y")
    f1 = x / y 
    f2 = y / x
    f3 = f1
    f3 /= x
    assert f1.diff() == {'x': 1/3, 'y': -2/3}, "Error: x/y, false derivative."
    assert f1.val == 2, "Error: x/y, false value."

    assert f2.diff() == {'x': -1/12, 'y': 1/6}, "Error: y/x, false derivative."
    assert f2.val == 1/2, "Error: y/x, false value."

    assert f3.diff() == {'x': 0, 'y': -1/9}, "Error: x/=y false derivative."
    assert f3.val == 1/3, "Error: x/=y, false value."

def test_pow_constant():
    x = AD.AD(2, "x")
    f1 = x ** 2 
    f2 = 2 ** f1
    f3 = f1
    f3 **= 2
    assert f1.diff("x") == 4, "Error: x**a, false derivative."
    assert f1.val == 4, "Error: x**a, false value {f1}."

    assert f2.diff("x") == math.log(2) * (2**4) * 4, "Error: a**x, false derivative."
    assert f2.val == 16, "Error: a**x, false value."

    assert f3.diff("x") == 4*(2**3), "Error: x**=a false derivative."
    assert f3.val == 16, "Error: f**=a, false value."

def test_pow_variable():
    x = AD.AD(2, "x")
    y = AD.AD(3, "y")
    f1 = x ** y 
    f2 = y ** x
    f3 = f1
    f3 **= x
    assert f1.diff() == {'x': 12, 'y': math.log(2) * (2**3)}, "Error: x**y, false derivative."
    assert f1.val == 8, "Error: x**y, false value."

    assert f2.diff() == {'x':  math.log(3) * (3**2), 'y': 6}, "Error: y**x, false derivative." 
    assert f2.val == 9, "Error: y**x, false value." 

    assert f3.diff() == {'x': (math.log(2)+1) * (2**6) * 3, 'y': math.log(2) * (2**6) * 2}, "Error: x**=y false derivative." 
    assert f3.val == 64, "Error: f**=x, false value." 

def test_jacobian():
    x = AD.AD(2, "x")
    y = AD.AD(3, "y")
    
    f = x/x  
    assert AD.jacobian(f) == [0], "Error: x/x, false jacobian."
    
    f = x**y
    f += y 
    assert AD.jacobian(f) == [12, math.log(2) * (2**3)+1], "Error: x**y + y, false jacobian."
    
    f = x/y
    assert AD.jacobian(f) == [1/3, -2/3**2], "Error: x/y, false jacobian."
    

    f = (math.e ** y)/x
    assert AD.jacobian(f) == [-math.e**3 / 4, math.e**3 /2], "Error: e**y /x, false jacobian."

def test_chain_rule():
    ad = AD.AD(2, "x")
    der = 2
    new_val = 3
    assert chain_rule(ad, new_val, der).__eq__(AD.AD(3, "x", 2))

def test_abs():
    x = AD.AD(1, "x")
    f = abs(x)
    assert f.val == 1
    assert f.ders == {'x': 1}
    x = AD.AD(-1, "x")
    f = abs(x)
    assert f.val == 1
    assert f.ders == {'x': -1}
    x = AD.AD(0, "x")
    with pytest.raises(Exception):
        f = abs(x)



def test_log():
    x = AD.AD(1, "x")
    g = log(x)
    assert g.ders == {'x': 1}
    assert g.val == math.log(1)
    assert g.tags == ['x']


# trig test:
def test_sin_float():
    ad = 2
    assert sin(ad).__eq__(math.sin(2)), "Error: sin(x), false value."

def test_sin_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert sin(x+y).val == math.sin(3), "Error: sin(x+y), false value."
    assert sin(x+y).diff() == {'x': math.cos(3), 'y': math.cos(3)}, "Error: sin(x+y), false derivative."

def test_cos_float():
    ad = 2
    assert cos(ad).__eq__(math.cos(2)), "Error: cos(x), false value."

def test_cos_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert cos(x+y).val == math.cos(3), "Error: cos(x+y), false value."
    assert cos(x+y).diff() == {'x': -math.sin(3), 'y': -math.sin(3)}, "Error: cos(x+y), false derivative."

def test_tan_float():
    ad = 2
    assert tan(ad).__eq__(math.tan(2)), "Error: tan(x), false value."

def test_tan_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert tan(x+y).val == math.tan(3), "Error: tan(x+y), false value."
    assert tan(x+y).diff() == {'x': 1/(math.cos(3)**2), 'y': 1/(math.cos(3)**2)}, "Error: tan(x+y), false derivative."

def test_cot_float():
    ad = 2
    assert cot(ad).__eq__(1/math.tan(2)), "Error: cot(x), false value."

epsilon = 10**(-10)

def test_cot_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert cot(x+y).val == 1/math.tan(3), "Error: cot(x+y), false value."
    assert abs(cot(x+y).diff('x') - (-1/(math.sin(3)**2))) <= epsilon, "Error: cot(x+y), false derivative."


def test_sec_float():
    ad = 2
    assert abs(sec(ad)-1/math.cos(2)) <= epsilon, "Error: sec(x), false value."

def test_sec_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert sec(x+y).val == 1/math.cos(3), "Error: sec(x+y), false value."
    assert sec(x+y).diff('x') - (math.tan(3)/math.cos(3)) <= epsilon, "Error: sec(x+y), false derivative."

def test_csc_float():
    ad = 2
    assert abs(csc(ad)-1/math.sin(2)) <= epsilon, "Error: csc(x), false value."

def test_sec_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert abs(csc(x+y).val - 1/math.sin(3)) <= epsilon, "Error: csc(x+y), false value."
    assert abs(csc(x+y).diff('x') - (-1/(math.tan(3)*math.sin(3)))) <= epsilon, "Error: csc(x+y), false derivative."

# hyperbolic trig
def test_sinh_float():
    ad = 2
    assert abs(sinh(ad)-math.sinh(2)) <= epsilon, "Error: sinh(x), false value."

def test_sinh_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert abs(sinh(x+y).val - math.sinh(3)) <= epsilon, "Error: sinh(x+y), false value."
    assert abs(sinh(x+y).diff('x') - math.cosh(3)) <= epsilon, "Error: sinh(x+y), false derivative."

def test_cosh_float():
    ad = 2
    assert abs(cosh(ad)-math.cosh(2)) <= epsilon, "Error: cosh(x), false value."

def test_cosh_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert abs(cosh(x+y).val - math.cosh(3)) <= epsilon, "Error: cosh(x+y), false value."
    assert abs(cosh(x+y).diff('x') - math.sinh(3)) <= epsilon, "Error: sinh(x+y), false derivative."

def test_tanh_float():
    ad = 2
    assert abs(tanh(ad)-math.tanh(2)) <= epsilon, "Error: tanh(x), false value."

def test_tanh_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert abs(tanh(x+y).val - math.tanh(3)) <= epsilon, "Error: tanh(x+y), false value."
    assert abs(tanh(x+y).diff('x') - 1/(math.cosh(3)**2)) <= epsilon, "Error: tanh(x+y), false derivative."

def test_coth_float():
    ad = 2
    assert abs(coth(ad)-math.cosh(2)/math.sinh(2)) <= epsilon, "Error: coth(x), false value."

def test_coth_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert abs(coth(x+y).val - math.cosh(3)/math.sinh(3)) <= epsilon, "Error: coth(x+y), false value."
    assert abs(coth(x+y).diff('x') - (-1/(math.sinh(3)**2))) <= epsilon, "Error: coth(x+y), false derivative."

def test_sech_float():
    ad = 2
    assert abs(sech(ad)-1/math.cosh(2)) <= epsilon, "Error: sech(x), false value."

def test_sech_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert abs(sech(x+y).val - 1/math.cosh(3)) <= epsilon, "Error: sech(x+y), false value."
    assert abs(sech(x+y).diff('x') - (-math.tanh(3)/math.cosh(3))) <= epsilon, "Error: sech(x+y), false derivative."

def test_csch_float():
    ad = 2
    assert abs(csch(ad)-1/math.sinh(2)) <= epsilon, "Error: csch(x), false value."

def test_csch_ad():
    x = AD.AD(1, "x")
    y = AD.AD(2, "y")
    assert abs(csch(x+y).val - 1/math.sinh(3)) <= epsilon, "Error: csch(x+y), false value."
    assert abs(csch(x+y).diff('x') - 1/(-math.sinh(3)*math.tanh(3))) <= epsilon, "Error: csch(x+y), false derivative."


# trig inverse
def test_asin_float():
    ad = 1/2
    assert abs(asin(ad)-math.asin(1/2)) <= epsilon, "Error: asin(x), false value."

def test_asin_ad():
    x = AD.AD(1/2, "x")
    y = AD.AD(1/3, "y")
    assert abs(asin(x+y).val - math.asin(5/6)) <= epsilon, "Error: asin(x+y), false value."
    assert abs(asin(x+y).diff('x') - 1/math.sqrt(1-25/36)) <= epsilon, "Error: asin(x+y), false derivative."

def test_acos_float():
    ad = 1/2
    assert abs(acos(ad)-math.acos(1/2)) <= epsilon, "Error: acos(x), false value."

def test_asin_ad():
    x = AD.AD(1/2, "x")
    y = AD.AD(1/3, "y")
    assert abs(acos(x+y).val - math.acos(5/6)) <= epsilon, "Error: acos(x+y), false value."
    assert abs(acos(x+y).diff('x') + 1/math.sqrt(1-25/36)) <= epsilon, "Error: acos(x+y), false derivative."

def test_atan_float():
    ad = 1/2
    assert abs(atan(ad)-math.atan(1/2)) <= epsilon, "Error: atan(x), false value."

def test_atan_ad():
    x = AD.AD(1/2, "x")
    y = AD.AD(1/3, "y")
    assert abs(atan(x+y).val - math.atan(5/6)) <= epsilon, "Error: atan(x+y), false value."
    assert abs(atan(x+y).diff('x') - 1/(1+25/36)) <= epsilon, "Error: atan(x+y), false derivative."

def test_acot_float():
    ad = 1/2
    assert abs(acot(ad)-math.atan(2)) <= epsilon, "Error: acot(x), false value."

def test_acot_ad():
    x = AD.AD(1/2, "x")
    y = AD.AD(1/3, "y")
    assert abs(acot(x*y).val - math.atan(6)) <= epsilon, "Error: acot(x*y), false value."
    assert abs(acot(x*y).diff('x') + (1/3) * 1/(1+1/36)) <= epsilon, "Error: acot(x*y), false derivative."

def test_asec_float():
    ad = 2
    assert abs(asec(ad)-math.acos(1/2)) <= epsilon, "Error: asec(x), false value."

def test_asec_ad():
    x = AD.AD(2, "x")
    y = AD.AD(3, "y")
    assert abs(asec(x*y).val - math.acos(1/6)) <= epsilon, "Error: asec(x*y), false value."
    assert abs(asec(x*y).diff('x') - 3/(6*math.sqrt(36-1))) <= epsilon, "Error: asec(x*y), false derivative."

def test_acsc_float():
    ad = 2
    assert abs(acsc(ad)-math.asin(1/2)) <= epsilon, "Error: acsc(x), false value."

def test_acsc_ad():
    x = AD.AD(2, "x")
    y = AD.AD(3, "y")
    assert abs(acsc(x*y).val - math.asin(1/6)) <= epsilon, "Error: acsc(x*y), false value."
    assert abs(acsc(x*y).diff('x') + 3/(6*math.sqrt(36-1))) <= epsilon, "Error: acsc(x*y), false derivative."


# hyperbolic trig inverse
def test_asinh_float():
    ad = 1/2
    assert abs(asinh(ad)-math.asinh(1/2)) <= epsilon, "Error: asinh(x), false value."

def test_asinh_ad():
    x = AD.AD(1/2, "x")
    y = AD.AD(1/3, "y")
    assert abs(asinh(x+y).val - math.asinh(5/6)) <= epsilon, "Error: asinh(x+y), false value."
    assert abs(asinh(x+y).diff('x') - 1/math.sqrt(25/36+1)) <= epsilon, "Error: asinh(x+y), false derivative."

def test_acosh_float():
    ad = 2
    assert abs(acosh(ad)-math.acosh(2)) <= epsilon, "Error: acosh(x), false value."

def test_acosh_ad():
    x = AD.AD(2, "x")
    y = AD.AD(3, "y")
    assert abs(acosh(x+y).val - math.acosh(5)) <= epsilon, "Error: acosh(x+y), false value."
    assert abs(acosh(x+y).diff('x') - 1/math.sqrt(25-1)) <= epsilon, "Error: acosh(x+y), false derivative."

def test_atanh_float():
    ad = 1/2
    assert abs(atanh(ad)-math.atanh(1/2)) <= epsilon, "Error: atanh(x), false value."

def test_atanh_ad():
    x = AD.AD(1/2, "x")
    y = AD.AD(1/3, "y")
    assert abs(atanh(x+y).val - math.atanh(5/6)) <= epsilon, "Error: atanh(x+y), false value."
    assert abs(atanh(x+y).diff('x') - 1/(1-25/36)) <= epsilon, "Error: atanh(x+y), false derivative."

def test_acoth_float():
    ad = 2
    assert abs(acoth(ad)-0.5*math.log((ad+1)/(ad-1))) <= epsilon, "Error: acoth(x), false value."

def test_acoth_ad():
    x = AD.AD(2, "x")
    y = AD.AD(3, "y")
    assert abs(acoth(x*y).val - 0.5*math.log((6+1)/(6-1))) <= epsilon, "Error: acoth(x*y), false value."
    assert abs(acoth(x*y).diff('x') - 3 * 1/(1-36)) <= epsilon, "Error: acoth(x*y), false derivative."

def test_asech_float():
    ad = 1/2
    assert abs(asech(ad)-math.log((1+math.sqrt(1-ad**2))/ad)) <= epsilon, "Error: asech(x), false value."

def test_asech_ad():
    x = AD.AD(1/2, "x")
    y = AD.AD(1/3, "y")
    assert abs(asech(x*y).val - math.log((1+math.sqrt(1-1/36))/(1/6))) <= epsilon, "Error: asech(x*y), false value."
    assert abs(asech(x*y).diff('x') + (1/3)/((1/6)*math.sqrt(1-1/36))) <= epsilon, "Error: asech(x*y), false derivative."

def test_acsch_float():
    ad = 2
    assert abs(acsch(ad)-math.log(1/2+math.sqrt(1/4+1))) <= epsilon, "Error: acsch(x), false value."

def test_acsch_ad():
    x = AD.AD(2, "x")
    y = AD.AD(3, "y")
    assert abs(acsch(x*y).val - math.log(1/6+math.sqrt(1/36+1))) <= epsilon, "Error: acsch(x*y), false value."
    assert abs(acsch(x*y).diff('x') + 3/(6*math.sqrt(1+36))) <= epsilon, "Error: acsch(x*y), false derivative."

