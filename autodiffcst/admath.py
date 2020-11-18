import math
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import autodiffcst.AD as AD
# import AD as AD

# need chain_rule function
# from Runting


def chain_rule(ad, new_val, der):
    """
    Applies chain rule to returns a new AD object with correct value and derivatives.

            Parameters:
                    ad (AD): An AD object
                    new_val (float): Value of the new AD object
                    der (float): Derivative of the outer function in chain rule

            Returns:
                    new_ad (AD): a new AD object with correct value and derivatives
    """
    new_ders = dict()
    for tag in ad.tags:
        new_ders[tag] = der * ad.ders[tag]
    new_ad = AD.AD(new_val, ad.tags, new_ders)
    return new_ad


# all functions could take either AD object or a number as input
# will raise TypeError with other inputs 

def abs(ad):
    """
    Returns the new AD object after applying absolute value function.

            Parameters:
                    ad (AD): An AD object to be applied absolute value function on

            Returns:
                    new_ad (AD): the new AD object after applying absolute value function
    """
    if isinstance(ad, AD.AD):
        new_val = math.fabs(ad.val)
        if ad.val > 0:
            der = 1
        elif ad.val < 0:
            der = -1
        else:
            raise Exception("Derivative undefined")
        return chain_rule(ad, new_val, der)
    else:
        return math.fabs(ad)
    

def exp(ad):
    """
    Returns the new AD object after applying exponential function.

            Parameters:
                    ad (AD): An AD object to be applied exponential function on

            Returns:
                    new_ad (AD): the new AD object after applying exponential function
    """
    if isinstance(ad, AD.AD):
        new_val = math.exp(ad.val)
        der = new_val
        return chain_rule(ad, new_val, der)
    else:
        return math.exp(ad)


def log(ad): #consider different base?
    """
    Returns the new AD object after applying log(base e) function.

            Parameters:
                    ad (AD): An AD object to be applied log(base e) function on

            Returns:
                    new_ad (AD): the new AD object after applying log(base e) function
    """
    if isinstance(ad, AD.AD):
        new_val = math.log(ad.val)
        der = 1/ad.val
        return chain_rule(ad, new_val, der)
    else:
        return math.log(ad)

def pow(ad, y):
    """
    Returns the new AD object after applying power function with power y.

            Parameters:
                    ad (AD): An AD object to be applied power function with power y on

            Returns:
                    new_ad (AD): the new AD object after applying power function with power y
    """
    if isinstance(ad, AD.AD):
        new_val = math.pow(ad.val, y)
        der = y * math.pow(ad.val, y - 1)
        return chain_rule(ad, new_val, der)
    else:
        return math.pow(ad, y)

def sqrt(ad):
    """
    Returns the new AD object after applying square root function.

            Parameters:
                    ad (AD): An AD object to be applied square root function on

            Returns:
                    new_ad (AD): the new AD object after applying square root function
    """
    return ad ** 0.5

# trig
def sin(ad):
    """
    Returns the new AD object after applying sine function.

            Parameters:
                    ad (AD): An AD object to be applied sine function on

            Returns:
                    new_ad (AD): the new AD object after applying sine function
    """
    if isinstance(ad, AD.AD):
        new_val = sin(ad.val)
        der = cos(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.sin(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def cos(ad):
    """
    Returns the new AD object after applying cosine function.

            Parameters:
                    ad (AD): An AD object to be applied cosine function on

            Returns:
                    new_ad (AD): the new AD object after applying cosine function
    """
    if isinstance(ad, AD.AD):
        new_val = cos(ad.val)
        der = -sin(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.cos(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def tan(ad):
    """
    Returns the new AD object after applying tangent function.

            Parameters:
                    ad (AD): An AD object to be applied tangent function on

            Returns:
                    new_ad (AD): the new AD object after applying tangent function
    """
    if isinstance(ad, AD.AD):
        new_val = tan(ad.val)
        der = sec(ad.val)**2
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.tan(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def cot(ad):
    """
    Returns the new AD object after applying cotangent function.

            Parameters:
                    ad (AD): An AD object to be applied cotangent function on

            Returns:
                    new_ad (AD): the new AD object after applying cotangent function
    """
    if isinstance(ad, AD.AD):
        new_val = cot(ad.val)
        der = -csc(ad.val)**2
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return 1/math.tan(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def sec(ad):
    """
    Returns the new AD object after applying secant function.

            Parameters:
                    ad (AD): An AD object to be applied secant function on

            Returns:
                    new_ad (AD): the new AD object after applying secant function
    """
    if isinstance(ad, AD.AD):
        new_val = sec(ad.val)
        der = sec(ad.val)*tan(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return 1/math.cos(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def csc(ad):
    """
    Returns the new AD object after applying cosecant function.

            Parameters:
                    ad (AD): An AD object to be applied cosecant function on

            Returns:
                    new_ad (AD): the new AD object after applying cosecant function
    """
    if isinstance(ad, AD.AD):
        new_val = csc(ad.val)
        der = -csc(ad.val)*cot(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return 1/math.sin(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")



# hyperbolic trig
def sinh(ad):
    """
    Returns the new AD object after applying hyperbolic sine function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic sine function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic sine function
    """
    if isinstance(ad, AD.AD):
        new_val = sinh(ad.val)
        der = cosh(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.sinh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def cosh(ad):
    """
    Returns the new AD object after applying hyperbolic cosine function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic cosine function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic cosine function
    """
    if isinstance(ad, AD.AD):
        new_val = cosh(ad.val)
        der = sinh(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.cosh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def tanh(ad):
    """
    Returns the new AD object after applying hyperbolic tangent function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic tangent function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic tangent function
    """
    if isinstance(ad, AD.AD):
        new_val = tanh(ad.val)
        der = sech(ad.val)**2
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.tanh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def coth(ad):
    """
    Returns the new AD object after applying hyperbolic cotangent function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic cotangent function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic cotangent function
    """
    if isinstance(ad, AD.AD):
        new_val = coth(ad.val)
        der = -csch(ad.val)**2
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.cosh(ad)/math.sinh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def sech(ad):
    """
    Returns the new AD object after applying hyperbolic secant function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic secant function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic secant function
    """
    if isinstance(ad, AD.AD):
        new_val = sech(ad.val)
        der = -sech(ad.val)*tanh(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return 1/math.cosh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def csch(ad):
    """
    Returns the new AD object after applying hyperbolic cosecant function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic cosecant function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic cosecant function
    """
    if isinstance(ad, AD.AD):
        new_val = csch(ad.val)
        der = -csch(ad.val)*coth(ad.val)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return 1/math.sinh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")




# inverse trig
def asin(ad):
    """
    Returns the new AD object after applying arc sine function.

            Parameters:
                    ad (AD): An AD object to be applied arc sine function on

            Returns:
                    new_ad (AD): the new AD object after applying arc sine function
    """
    if isinstance(ad, AD.AD):
        new_val = asin(ad.val)
        der = 1/math.sqrt(1-ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.asin(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acos(ad):
    """
    Returns the new AD object after applying arc cosine function.

            Parameters:
                    ad (AD): An AD object to be applied arc cosine function on

            Returns:
                    new_ad (AD): the new AD object after applying arc cosine function
    """
    if isinstance(ad, AD.AD):
        new_val = acos(ad.val)
        der = -1/math.sqrt(1-ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.acos(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def atan(ad):
    """
    Returns the new AD object after applying arc tangent function.

            Parameters:
                    ad (AD): An AD object to be applied arc tangent function on

            Returns:
                    new_ad (AD): the new AD object after applying arc tangent function
    """
    if isinstance(ad, AD.AD):
        new_val = atan(ad.val)
        der = 1/(1+ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.atan(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acot(ad):
    """
    Returns the new AD object after applying arc cotangent function.

            Parameters:
                    ad (AD): An AD object to be applied arc cotangent function on

            Returns:
                    new_ad (AD): the new AD object after applying arc cotangent function
    """
    if isinstance(ad, AD.AD):
        new_val = acot(ad.val)
        der = -1/(1+ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.atan(1/ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def asec(ad):
    """
    Returns the new AD object after applying arc secant function.

            Parameters:
                    ad (AD): An AD object to be applied arc secant function on

            Returns:
                    new_ad (AD): the new AD object after applying arc secant function
    """
    if isinstance(ad, AD.AD):
        new_val = asec(ad.val)
        if abs(ad.val) <= 1:
            raise ValueError("To be differentiable, asec cannot take input within (-1,1).")
        else:
            der = 1/(abs(ad.val)*math.sqrt(ad.val**2-1))
            return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.acos(1/ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acsc(ad):
    """
    Returns the new AD object after applying arc cosecant function.

            Parameters:
                    ad (AD): An AD object to be applied arc cosecant function on

            Returns:
                    new_ad (AD): the new AD object after applying arc cosecant function
    """
    if isinstance(ad, AD.AD):
        new_val = acsc(ad.val)
        if abs(ad.val) <= 1:
            raise ValueError("To be differentiable, acsc cannot take input within (-1,1).")
        else:
            der = -1/(abs(ad.val)*math.sqrt(ad.val**2-1))
            return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.asin(1/ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")
    


# inverse hyperbolic trig
def asinh(ad):
    """
    Returns the new AD object after applying hyperbolic arc sine function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic arc sine function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic arc sine function
    """
    if isinstance(ad, AD.AD):
        new_val = asinh(ad.val)
        der = 1/math.sqrt(1+ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        return math.asinh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acosh(ad):
    """
    Returns the new AD object after applying hyperbolic arc cosine function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic arc cosine function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic arc cosine function
    """
    if isinstance(ad, AD.AD):
        new_val = acosh(ad.val)
        der = 1/math.sqrt(ad.val**2-1)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        if ad < 1:
            raise ValueError("The domain of acosh is [1,infty).")
        else:
            return math.acosh(ad)
    else:
        raise TypeError("Input should be either an AD object or a number.")

def atanh(ad):
    """
    Returns the new AD object after applying hyperbolic arc tangent function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic arc tangent function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic arc tangent function
    """
    if isinstance(ad, AD.AD):
        new_val = atanh(ad.val)
        der = 1/(1-ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        if ad > -1 and ad < 1:
            return math.atanh(ad)
        else:
            raise ValueError("The domain of atanh is (-1,1).")
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acoth(ad):
    """
    Returns the new AD object after applying hyperbolic arc cotangent function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic arc cotangent function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic arc cotangent function
    """
    if isinstance(ad, AD.AD):
        new_val = acoth(ad.val)
        der = 1/(1-ad.val**2)
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        if -1 <= ad and ad <= 1:
            raise ValueError("The domain of acoth is (-infty,-1)U(1,infty).")
        else:
            return 0.5*math.log((ad+1)/(ad-1))
    else:
        raise TypeError("Input should be either an AD object or a number.")

def asech(ad):
    """
    Returns the new AD object after applying hyperbolic arc secant function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic arc secant function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic arc secant function
    """
    if isinstance(ad, AD.AD):
        new_val = asech(ad.val)
        der = -1/(ad.val*(ad.val+1)*math.sqrt((1-ad.val)/(1+ad.val)))
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        if ad > 0 and ad <= 1: 
            return math.log((1+math.sqrt(1-ad**2))/ad)
        else:
            raise ValueError("The domain of asech is (0,1].")
    else:
        raise TypeError("Input should be either an AD object or a number.")

def acsch(ad):
    """
    Returns the new AD object after applying hyperbolic arc cosecant function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic arc cosecant function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic arc cosecant function
    """
    if isinstance(ad, AD.AD):
        new_val = acsch(ad.val)
        der = -1/(ad.val**2*math.sqrt(1/ad.val**2+1))
        return chain_rule(ad, new_val, der)
    elif isinstance(ad, int) or isinstance(ad, float):
        if ad == 0:
            raise ValueError("The domain of acsch is (-infty,0)U(0,infty).")
        else:
            return math.log(1/ad+math.sqrt(1/(ad**2)+1))
    else:
        raise TypeError("Input should be either an AD object or a number.")
