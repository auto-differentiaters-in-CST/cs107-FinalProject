import math
import numbers
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import sympy as sp
import autodiffcst.AD as AD
import autodiffcst.AD_vec as VAD


def set_VAD(ADs):
    """
    Uses the information of new ADs to generate a new VAD

            Parameters:
                    ADs: An array of AD objects
            
            Returns:
                    A new VAD object, which has all the variables and their val, der, der2
    """
    new_val = np.array([ADs[i].val for i in range(len(ADs))])
    new_der = np.array([ADs[i].der for i in range(len(ADs))])
    new_der2 = np.array([ADs[i].der2 for i in range(len(ADs))])
    return VAD.VAD(new_val, new_der, new_der2)


def chain_rule(ad, new_val, der, der2, higher_der = None):
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
        new_higher_der = np.array([0.0]*len(ad.higher))
        new_higher_der[0] = new_der
        new_higher_der[1] = new_der2
        for i in range(2, len(ad.higher)):
            n = i+1
            sum = 0
            for k in range(1,n+1):
                sum += higher_der[k-1]*sp.bell(n, k, ad.higher[0:n-k+1])
            new_higher_der[i] = sum
        new_ad = AD.AD(new_val,tag=ad.tag, der=new_der, der2=new_der2, order = len(ad.higher))
        new_ad.higher = new_higher_der

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
        new_val = np.abs((ad.val))
        der = np.array([0]*len(ad))
        der2 = np.array([0]*len(ad))
        def get_der(v):
            if v > 0:
                return 1
            elif v < 0:
                return -1
            else:
                raise Exception("Derivative undefined")
        for i in range(0,len(ad.val)):
            if isinstance(ad.val[i], numbers.Integral):
                der[i] = get_der(ad.val[i])
            else:
                print(type(ad.val[i]))
                sub_der = np.array([get_der(v) for v in ad.val[i]])
                der[i] = sub_der
                der2[i] = np.array([0]*len(ad.val[i]))

        if ad.higher is None:
            return chain_rule(ad, new_val, der, der2)
        else:
            higher_der = np.array([new_val]*len(ad.higher))
            #print(higher_der)
            return chain_rule(ad, new_val, der, der2, higher_der)
    elif isinstance(ad, VAD.VAD):
        AD_result = np.array([abs(advar) for advar in ad.variables])
        return set_VAD(AD_result)
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
            higher_der = np.array([new_val]*len(ad.higher))
            #print(higher_der)
            return chain_rule(ad, new_val, der, der2, higher_der)
    elif isinstance(ad, VAD.VAD):
        AD_result = np.array([exp(advar) for advar in ad.variables])
        return set_VAD(AD_result) 
    else:
        try:
            return np.exp(ad)
        except:
            raise TypeError("Your input is not valid.")




def log(ad): #consider different base?
    """
    Returns the new AD object after applying log(base e) function.

            Parameters:
                    ad (AD): An AD object to be applied log(base e) function on

            Returns:
                    new_ad (AD): the new AD object after applying log(base e) function
    """
    if isinstance(ad, AD.AD):
        new_val = np.log(ad.val)
        der = 1/ad.val
        der2 = -1/ad.val**2
        return chain_rule(ad, new_val, der, der2)
    else:
        return np.log(ad)

def fact_ad(x,n):
    """
    Returns x(x-1)(x-2)...(x-n+1), the product of n terms, factorial-like operation
            Parameters:
                    x, n: two scalars
            Returns:
                    x(x-1)(x-2)...(x-n+1): scalar
    """
    prod = 1
    for i in range(n):
        prod = prod * (x-i)
    return prod
    

def pow(ad, y):
    """
    Returns the new AD object after applying power function with power y.
            Parameters:
                    ad (AD): An AD object to be applied power function with power y on
            Returns:
                    new_ad (AD): the new AD object after applying power function with power y
    """
    if isinstance(y, AD.AD):
        raise TypeError("Error: y cannot be an AD object.")
    if isinstance(ad, AD.AD):
        new_val = np.pow(ad.val, y)
        der = y * np.pow(ad.val, y - 1)
        der2 = y * (y-1) * np.pow(ad.val, y-2)
        if ad.higher is None:
            return chain_rule(ad, new_val, der, der2)
        else:
            higher_der = np.array([0.0]*len(ad.higher))
            for i in range(len(ad.higher)):
                n = i + 1
                # derivative d(i+1) = y(y-1)...(y-n+1) ad.val **(y-i-1)
                # for example: f = x**6, d(1) = 6*x**5, d(2) = 6*5*x**4, d(3) = 6*5*4*x**3 
                # so for d(n)=d(i+1), the power after x is y-(i+1), 
                #                     the coef is y(y-1)...(y-i)=y(y-1)...(y-(n-1)) (product of n terms)
                higher_der[i] = np.pow(ad.val,y-i) * fact_ad(ad.val,n)
            print(higher_der)
            return chain_rule(ad, new_val, der, der2, higher_der)
    elif isinstance(ad, VAD.VAD):
        return ad**y 
    else:
        try:
            return np.power(ad,y)
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
            higher_der = np.array([der, der2, -der, -der2] * int(np.ceil(len(ad.higher)/4)))
            higher_der = higher_der[0:len(ad.higher)]
            return chain_rule(ad, new_val, der, der2, higher_der)
    elif isinstance(ad, VAD.VAD):
        AD_result = np.array([sin(advar) for advar in ad.variables])
        return set_VAD(AD_result) 
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
    elif isinstance(ad, VAD.VAD):
        AD_result = np.array([cos(advar) for advar in ad.variables])
        return set_VAD(AD_result) 
    else:
        try:
            return np.cos(ad)
        except:
            raise TypeError("Your input is not valid.")


def tan(ad): ## problem
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
        der2 = 2*tan(ad.val)*sec(ad.val)**2
        if ad.higher is None:
            return chain_rule(ad, new_val, der, der2)
        else:
            higher_der = np.array([der, der2, -der, -der2] * int(np.ceil(len(ad.higher) / 4)))
            higher_der = higher_der[0:len(ad.higher)]
            return chain_rule(ad, new_val, der, der2, higher_der)
    elif isinstance(ad, VAD.VAD):
        AD_result = np.array([tan(advar) for advar in ad.variables])
        return set_VAD(AD_result) 
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
        return 1/np.cos(num)
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
            higher_der = np.array([der, der2, der, der2] * int(np.ceil(len(ad.higher)/4)))
            higher_der = higher_der[0:len(ad.higher)]
            return chain_rule(ad, new_val, der, der2, higher_der)
    elif isinstance(ad, VAD.VAD):
        AD_result = np.array([sinh(advar) for advar in ad.variables])
        return set_VAD(AD_result) 
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
            higher_der = np.array([der, der2, der, der2] * int(np.ceil(len(ad.higher)/4)))
            higher_der = higher_der[0:len(ad.higher)]
            return chain_rule(ad, new_val, der, der2, higher_der)
    elif isinstance(ad, VAD.VAD):
        AD_result = np.array([cosh(advar) for advar in ad.variables])
        return set_VAD(AD_result) 
    else:
        try:
            return np.cosh(ad)
        except:
            raise TypeError("Your input is not valid.")


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