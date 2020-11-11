import math

# need chain_rule function
# from Runting
def chain_rule(ad, new_val, der):
    new_ders = dict()
    for tag in ad.tags:
        new_ders[tag] = der * ad.ders[tag]
    new_ad = AD(new_val, ad.tags, new_ders)
    return new_ad


# all functions could take either AD object or a number as input
# will raise TypeError with other inputs 



# trig
def sin(ad):
    if isinstance(ad, AD):
        new_val = sin(ad.val)
        der = cos(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.sin(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def cos(ad):
    if isinstance(ad, AD):
        new_val = cos(ad.val)
        der = -sin(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.cos(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def tan(ad):
    if isinstance(ad, AD):
        new_val = tan(ad.val)
        der = sec(ad.val)**2
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.tan(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def cot(ad):
    if isinstance(ad, AD):
        new_val = cot(ad.val)
        der = sec(ad.val)**2
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return 1/math.tan(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def sec(ad):
    if isinstance(ad, AD):
        new_val = sec(ad.val)
        der = sec(ad.val)*tan(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return 1/math.cos(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def csc(ad):
    if isinstance(ad, AD):
        new_val = csc(ad.val)
        der = csc(ad.val)*cot(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return 1/math.sin(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")



# hyperbolic trig
def sinh(ad):
    if isinstance(ad, AD):
        new_val = sinh(ad.val)
        der = cosh(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.sinh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def cosh(ad):
    if isinstance(ad, AD):
        new_val = cosh(ad.val)
        der = sinh(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.cosh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def tanh(ad):
    if isinstance(ad, AD):
        new_val = tanh(ad.val)
        der = sech(ad.val)**2
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.tanh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def coth(ad):
    if isinstance(ad, AD):
        new_val = coth(ad.val)
        der = csch(ad.val)**2
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.cosh(ad)/math.sinh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def sech(ad):
    if isinstance(ad, AD):
        new_val = sech(ad.val)
        der = sech(ad.val)*tanh(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return 1/math.cosh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def csch(ad):
    if isinstance(ad, AD):
        new_val = csch(ad.val)
        der = csch(ad.val)*coth(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return 1/math.sinh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")




# inverse trig
def asin(ad):
    if isinstance(ad, AD):
        new_val = asin(ad.val)
        der = 1/math.sqrt(1-ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.asin(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acos(ad):
    if isinstance(ad, AD):
        new_val = acos(ad.val)
        der = -1/math.sqrt(1-ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.acos(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def atan(ad):
    if isinstance(ad, AD):
        new_val = atan(ad.val)
        der = 1/(1+ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.atan(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acot(ad):
    if isinstance(ad, AD):
        new_val = acot(ad.val)
        der = -1/(1+ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.atan(1/ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def asec(ad):
    if isinstance(ad, AD):
        new_val = asec(ad.val)
        der = 1/(ad.val*(ad.val**2-1))
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.acos(1/ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acsc(ad):
    if isinstance(ad, AD):
        new_val = acsc(ad.val)
        der = -1/(ad.val*(ad.val**2-1))
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.asin(1/ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")
    


# inverse hyperbolic trig
def asinh(ad):
    if isinstance(ad, AD):
        new_val = asinh(ad.val)
        der = 1/math.sqrt(1+ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.asinh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acosh(ad):
    if isinstance(ad, AD):
        new_val = acosh(ad.val)
        der = 1/math.sqrt(ad.val**2-1)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.acosh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def atanh(ad):
    if isinstance(ad, AD):
        new_val = atanh(ad.val)
        der = 1/(1-ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.atanh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acoth(ad):
    if isinstance(ad, AD):
        new_val = acoth(ad.val)
        der = 1/(1-ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return 0.5*math.log((ad+1)/(ad-1))
    else:
        raise TypeError("Input should be either an AD object or a number.")

def asech(ad):
    if isinstance(ad, AD):
        new_val = asech(ad.val)
        der = -1/(ad.val*(ad.val+1)*math.sqrt((1-ad.val)/(1+ad.val)))
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.log((1+math.sqrt(1-ad**2))/ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acsch(ad):
    if isinstance(ad, AD):
        new_val = acsch(ad.val)
        der = -1/(ad.val**2*math.sqrt(1/ad.val**2+1))
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.log(1/ad+math.sqrt(1/(ad**2)+1))
    else:
        raise TypeError("Input should be either an AD object or a number.")
