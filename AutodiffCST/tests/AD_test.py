import pytest

import src.autodiffcst.AD as AD
import math



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
