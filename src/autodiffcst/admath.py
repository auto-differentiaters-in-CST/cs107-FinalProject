import math
import numbers
import os
import sys

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import sympy as sp
# import AD as AD
# import AD_vec as VAD
import autodiffcst.AD as AD



# def set_VAD(ADs):
#     """
#     Uses the information of new ADs to generate a new VAD
#
#             Parameters:
#                     ADs: An array of AD objects
#
#             Returns:
#                     A new VAD object, which has all the variables and their val, der, der2
#     """
#     new_val = np.array([ADs[i].val for i in range(len(ADs))])
#     new_der = np.array([ADs[i].der for i in range(len(ADs))])
#     new_der2 = np.array([ADs[i].der2 for i in range(len(ADs))])
#     return VAD.VAD(new_val, new_der, new_der2)


def choose(n, k):
    """
    A helper function that gives the value of n choose k, according to math definition

            Parameters:
                    n, k: both natural numbers, invalid input cases handled by numpy
            
            Returns:
                    the arithmetic value of n choose k, a scalar
    """
    return np.math.factorial(n) / (np.math.factorial(k) * np.math.factorial(n - k))


def chain_rule(ad, new_val, der, der2, higher_der=None):
    """
    Applies chain rule to returns a new AD object with correct value and derivatives.

            Parameters:
                    ad (AD): An AD object
                    new_val (float): Value of the new AD object
                    der (float): Derivative of the outer function in chain rule

            Returns:
                    new_ad (AD): a new AD object with correct value and derivatives
    """
    new_der = der * ad.der
    new_der2 = der * ad.der2 + der2 * np.matmul(np.array([ad.der]).T, np.array([ad.der]))
    if ad.higher is None:
        new_ad = AD.AD(new_val, tag=ad.tag, der=new_der, der2=new_der2)
    else:
        new_higher_der = np.array([0.0] * len(ad.higher))
        new_higher_der[0] = new_der
        new_higher_der[1] = new_der2
        for i in range(2, len(ad.higher)):
            n = i + 1
            sum = 0
            for k in range(1, n + 1):
                sum += higher_der[k - 1] * sp.bell(n, k, ad.higher[0:n - k + 1])
            new_higher_der[i] = sum
        new_ad = AD.AD(new_val, tag=ad.tag, der=new_der, der2=new_der2, order=len(ad.higher))
        new_ad.higher = new_higher_der

    return new_ad


# all functions could take either VAD, or AD object or a number/array/list (equivalent to numpy functions) as input

def abs(ad):
    """
    Returns the new AD object after applying absolute value function.

            Parameters:
                    ad (AD): An AD object to be applied absolute value function on

            Returns:
                    new_ad (AD): the new AD object after applying absolute value function
    """
    if isinstance(ad, AD.AD):
        new_val = np.abs((ad.val))
        der = np.array([0] * len(ad))
        der2 = np.array([0] * len(ad))

        def get_der(v):
            if v > 0:
                return 1
            elif v < 0:
                return -1
            else:
                raise Exception("Derivative undefined")

        for i in range(0, len(ad.val)):
            if isinstance(ad.val[i], numbers.Integral):
                der[i] = get_der(ad.val[i])
            else:
                print(type(ad.val[i]))
                sub_der = np.array([get_der(v) for v in ad.val[i]])
                der[i] = sub_der
                der2[i] = np.array([0] * len(ad.val[i]))

        if ad.higher is None:
            return chain_rule(ad, new_val, der, der2)
        else:
            higher_der = np.array([0.0] * len(ad.higher))
            higher_der[0] = der
            return chain_rule(ad, new_val, der, der2, higher_der)
    # elif isinstance(ad, VAD.VAD):
    #     AD_result = np.array([abs(advar) for advar in ad.variables])
    #     return set_VAD(AD_result)
    else:
        try:
            return np.abs(ad)
        except:
            raise TypeError("Your input is not valid.")


def exp(ad):
    """
    Returns the new AD object after applying exponential function.

            Parameters:
                    ad (AD): An AD object to be applied exponential function on

            Returns:
                    new_ad (AD): the new AD object after applying exponential function
    """
    if isinstance(ad, AD.AD):
        new_val = np.exp(ad.val)
        der = new_val
        der2 = new_val
        if ad.higher is None:
            return chain_rule(ad, new_val, der, der2)
        else:
            higher_der = np.array([new_val] * len(ad.higher))
            return chain_rule(ad, new_val, der, der2, higher_der)
    # elif isinstance(ad, VAD.VAD):
    #     AD_result = np.array([exp(advar) for advar in ad.variables])
    #     return set_VAD(AD_result)
    else:
        try:
            return np.exp(ad)
        except:
            raise TypeError("Your input is not valid.")


def fact_ad(x, n):
    """
    Returns x(x-1)(x-2)...(x-n+1), the product of n terms, factorial-like operation
            Parameters:
                    x, n: two scalars
            Returns:
                    x(x-1)(x-2)...(x-n+1): scalar
    """
    prod = 1
    for i in range(n):
        prod = prod * (x - i)
    return prod


def log(ad):  # consider different base?
    """
    Returns the new AD object after applying log(base e) function.

            Parameters:
                    ad (AD): An AD object to be applied log(base e) function on

            Returns:
                    new_ad (AD): the new AD object after applying log(base e) function
    """
    if isinstance(ad, AD.AD):
        new_val = np.log(ad.val)
        der = 1 / ad.val
        der2 = -1 / ad.val ** 2
        if ad.higher is None:
            return chain_rule(ad, new_val, der, der2)
        else:
            # starting from the first derivative: x**-1
            higher_der = np.array([0.0] * len(ad.higher))
            higher_der[0] = der
            higher_der[1] = der2
            for i in range(2, len(ad.higher)):
                n = i + 1
                coef = fact_ad(-1, n - 1)
                mainval = np.power(ad.val[0], float(-n))
                # mainval = math.pow(ad.val[0], -n)
                higher_der[i] = coef * mainval
            return chain_rule(ad, new_val, der, der2, higher_der)
    # elif isinstance(ad, VAD.VAD):
    #     AD_result = np.array([log(advar) for advar in ad.variables])
    #     return set_VAD(AD_result)
    else:
        try:
            return np.log(ad)
        except:
            raise TypeError("Your input is not valid.")





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
        der2 = -new_val
        if ad.higher is None:
            return chain_rule(ad, new_val, der, der2)
        else:
            higher_der = np.array([der, der2, -der, -der2] * int(np.ceil(len(ad.higher) / 4)))
            higher_der = higher_der[0:len(ad.higher)]
            return chain_rule(ad, new_val, der, der2, higher_der)
    # elif isinstance(ad, VAD.VAD):
    #     AD_result = np.array([sin(advar) for advar in ad.variables])
    #     return set_VAD(AD_result)
    else:
        try:
            return np.sin(ad)
        except:
            raise TypeError("Your input is not valid.")


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
        der2 = -cos(ad.val)
        if ad.higher is None:
            return chain_rule(ad, new_val, der, der2)
        else:
            higher_der = np.array([der, der2, -der, -der2] * int(np.ceil(len(ad.higher) / 4)))
            higher_der = higher_der[0:len(ad.higher)]
            return chain_rule(ad, new_val, der, der2, higher_der)
    # elif isinstance(ad, VAD.VAD):
    #     AD_result = np.array([cos(advar) for advar in ad.variables])
    #     return set_VAD(AD_result)
    else:
        try:
            return np.cos(ad)
        except:
            raise TypeError("Your input is not valid.")


def tan(ad):
    """
    Returns the new AD object after applying tangent function.

            Parameters:
                    ad (AD): An AD object to be applied tangent function on

            Returns:
                    new_ad (AD): the new AD object after applying tangent function
    """
    if isinstance(ad, AD.AD):
        return sin(ad) / cos(ad)
    # elif isinstance(ad, VAD.VAD):
    #     AD_result = np.array([tan(advar) for advar in ad.variables])
    #     return set_VAD(AD_result)
    else:
        try:
            return np.tan(ad)
        except:
            raise TypeError("Your input is not valid.")


# helper function for tan
def sec(num):
    """
    Returns the new AD object after applying secant function.
            Parameters:
                    num: a number or array
            Returns:
                    the result after applying secant function
    """
    try:
        return 1 / np.cos(num)
    except:
        raise TypeError("sec function can only handle number or array.")


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
        der2 = sinh(ad.val)
        if ad.higher is None:
            return chain_rule(ad, new_val, der, der2)
        else:
            higher_der = np.array([der, der2, der, der2] * int(np.ceil(len(ad.higher) / 4)))
            higher_der = higher_der[0:len(ad.higher)]
            return chain_rule(ad, new_val, der, der2, higher_der)
    # elif isinstance(ad, VAD.VAD):
    #     AD_result = np.array([sinh(advar) for advar in ad.variables])
    #     return set_VAD(AD_result)
    else:
        try:
            return np.sinh(ad)
        except:
            raise TypeError("Your input is not valid.")


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
        der2 = new_val
        if ad.higher is None:
            return chain_rule(ad, new_val, der, der2)
        else:
            higher_der = np.array([der, der2, der, der2] * int(np.ceil(len(ad.higher) / 4)))
            higher_der = higher_der[0:len(ad.higher)]
            return chain_rule(ad, new_val, der, der2, higher_der)
    # elif isinstance(ad, VAD.VAD):
    #     AD_result = np.array([cosh(advar) for advar in ad.variables])
    #     return set_VAD(AD_result)
    else:
        try:
            return np.cosh(ad)
        except:
            raise TypeError("Your input is not valid.")


def sech(ad):
    """
    Returns the new AD object after applying hyperbolic secant function.
            Parameters:
                    ad (AD): An AD object to be applied hyperbolic secant function on
            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic secant function
    """
    try:
        return 1 / np.cosh(ad)
    except:
        raise TypeError("sec function can only handle number or array.")


def tanh(ad):
    """
    Returns the new AD object after applying hyperbolic tangent function.

            Parameters:
                    ad (AD): An AD object to be applied hyperbolic tangent function on

            Returns:
                    new_ad (AD): the new AD object after applying hyperbolic tangent function
    """

    if isinstance(ad, AD.AD):
        return sinh(ad)/cosh(ad)
    # elif isinstance(ad, VAD.VAD):
    #     AD_result = np.array([tanh(advar) for advar in ad.variables])
    #     return set_VAD(AD_result)
    else:
        try:
            return np.tanh(ad)
        except:
            raise TypeError("Your input is not valid.")