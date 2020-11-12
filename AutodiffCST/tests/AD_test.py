import pytest

import src.autodiffcst.AD as AD
import math

#test1
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

def test_chain_rule():
    ad = AD.AD(2, "x")
    der = 2
    new_val = 3
    assert AD.chain_rule(ad, new_val, der).__eq__(AD.AD(3, "x", 2))

def test_abs():
    x = AD.AD(1, "x")
    f = AD.abs(x)
    assert f.val == 1
    assert f.ders == {'x': 1}
    x = AD.AD(-1, "x")
    f = AD.abs(x)
    assert f.val == 1
    assert f.ders == {'x': -1}
    x = AD.AD(0, "x")
    with pytest.raises(Exception):
        f = AD.abs(x)




def test_log():
    x = AD.AD(1, "x")
    g = AD.log(x)
    assert g.ders == {'x': 1}
    assert g.val == math.log(1)
    assert g.tags == ['x']
