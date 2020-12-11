import numpy as np
import numbers
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import autodiffcst.AD as ad
import autodiffcst.admath as admath


class VAD():

    def __init__(self, val, der=None, der2=None, order=2, higher=None):
        """
        Overwrites the __init__ dunder method to create a new VAD object with initial value and derivatives.
    
                Parameters:
                        val (int or float or list or np.array): the initial value of the new VAD object.
                        der (int or float or list or np.array): first-order derivatives of the new AD object. 
                        der2 (int or float or list or np.array): second-order derivatives of the new AD object. 

                Returns:
                        None, but initializes AD object(s) when called
                
                Example:
                >>> x, y = VAD([1,2]) 
                >>> x
                AD(value: [1], derivatives: [1., 0.])

                >>> fs = VAD([1,2]) 
                >>> fs
                VAD(value: [1 2], derivatives: [[1. 0.]
                                                [0. 1.]])
        """
        self.val = np.array(val)
        if der is None:
            self.der = np.eye(len(self))
            self.der2 = np.zeros((len(self),len(self),len(self)))
        else:
            self.der = der
            self.der2 = der2
        self.tag = np.array([i for i in range(len(self))])
        self.size = len(self)

        if not isinstance(order, numbers.Integral):
            raise TypeError("Highest order of derivatives must be a positive integer.")
        elif order < 1:
            raise ValueError("Highest order of derivatives must be at least 1.")
        elif order > 2 and self.size > 1:
            raise Exception("We cannot handle derivatives of order > 2 for more than one scalar variables.")

        self.order = order
        self.higher = higher

        if self.size > 1:
            arr_ad = np.array([None]*len(self))
            for i in range(len(self)):
                arr_ad[i] = ad.AD(val=self.val[i], tag=self.tag[i], size = self.size,
                                der=self.der[i], der2=self.der2[i])
            self.variables = arr_ad
        else:
            arr_ad = np.array([None]*len(self))
            arr_ad[0] = ad.AD(val=self.val[0], tag=self.tag[0], size = 1,
                                der=self.der[0], der2=self.der2[0], 
                                order=self.order, higher = self.higher)
            self.variables = arr_ad

    def __str__(self):
        """
        Overwrites the __str__ dunder method to nicely turn an AD object into a string.
    
                Parameters:
                        self (AD): the AD object that __str__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.
                
                Example:
                >>> fs = VAD([1,2]) 
                >>> print(fs)
                VAD(value: [1 2], derivatives: [[1. 0.]
                                                [0. 1.]])
        
        """
        return "VAD(value: {0}, derivatives: {1})".format(self.val, self.der)

    def __repr__(self):
        """
        Overwrites the __repr__ dunder method to nicely print an AD object.
    
                Parameters:
                        self (AD): the AD object that __repr__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.

                Example:
                >>> fs = VAD([1,2]) 
                >>> fs
                VAD(value: [1 2], derivatives: [[1. 0.]
                                                [0. 1.]])
        
        """
        return "VAD(value: {0}, derivatives: {1})".format(self.val, self.der)

    def __len__(self):
        """
        Overwrites the __len__ dunder method to get the length of the VAD object's values, namely the number of variables.
    
                Parameters:
                        self (VAD): the VAD object that __len__ is called upon.
    
                Returns:
                        An integer representing the length of the vector dimension that the VAD object resides in.

                Example:
                >>> len(VAD([1,2]))
                2
        """     

        try:
            return len(self.val)
        except TypeError:
            return 1

    ## getter
    def __getitem__(self, pos):
        """
        Overwrites the __getitem__ dunder method to get variable at certain position of the VAD object.
    
                Parameters:
                        self (VAD): the VAD object that __getitem__ is called upon.
                        pos (int): a valid position of a variable.
    
                Returns:
                        AD object at the position

                Example:
                >>> VAD([1,2])[0]
                AD(value: [1], derivatives: [1. 0.])
        """  

        return self.variables[pos]
        
    # Comparison Equal
    def __eq__(self, other):
        """
        Overwrites the __eq__ dunder method to check if two VAD objects are equal in value.
    
                Parameters:
                        self (VAD): the VAD object that __eq__ is called upon.
                        other (VAD): the VAD object to be compared with.
    
                Returns:
                        True if all the values of two VAD objects are equal.
                        False if otherwise.

                Example:
                >>> VAD([1,2]) == VAD([1,3])
                False
        """
        if isinstance(other, VAD):
            if np.sum(self.val == other.val) == len(self): 
                return True
                
            else:
                return False
        else:
            raise TypeError("Invalid Comparison. VAD object can only be compared with VAD.")

    def __ne__(self, other):
        """
        Overwrites the __eq__ dunder method to check if two VAD objects are equal in value.
    
                Parameters:
                        self (VAD): the VAD object that __ne__ is called upon.
                        other (VAD): the VAD object to be compared with.
    
                Returns:
                        False if all the values of two VAD objects are equal.
                        True if otherwise.

                Example:
                >>> VAD([1,2]) != VAD([1,3])
                True
        """
        return not self == other

    def isequal(self, other):
        """
        Compare value of two VAD objects' values element wise.

                Parameters:
                        self (VAD): the VAD object that isequal is called upon.
                        other (VAD): the VAD object to be compared with.
    
                Returns:
                        an array of True or False depends on the comparison result.
        
                Example:
                >>> a = VAD([1,2,3])
                >>> b = VAD([2,2,3])
                >>> a.isequal(b)
                array([False,True,True]) 
        """
        if isinstance(other, VAD):
            return self.val == other.val      
        else:
            raise TypeError("The input must also be a VAD object.")
    
    def fullequal(self, other):
        """
         Compare value, first derivatives and second derivatives of two VAD objects.

                Parameters:
                        self (VAD): the VAD object that fullequal is called upon.
                        other (VAD): the VAD object to be compared with.
    
                Returns:
                        True if everything is equal.
                        False otherwise.
     
                Example:
                >>> a = VAD([1,2,3])
                >>> b = VAD([1,2,3]) * 1
                >>> a.fullequal(b)
                True
        """
        if isinstance(other, VAD):
            return np.allclose(self.val, other.val) and np.allclose(self.der,other.der) and np.allclose(self.der2, other.der2)
        else:
            raise TypeError("Invalid Comparison. VAD object can only be compared with VAD.")


    def __gt__(self, other):
        """
        Overwrites the __gt__ dunder method to compare value of two VAD objects.

                Parameters:
                        self (VAD): the VAD object that __gt__ is called upon.
                        other (VAD): the VAD object to be compared with.
    
                Returns:
                        True if self.val is larger than other.val elementwise.
                        False otherwise.
                Example:
                >>> a = VAD([2,2,3])
                >>> b = VAD([1,2,2])
                >>> a > b
                False
        """
        if isinstance(other, VAD):
            return np.sum(self.val > other.val) == len(self)
        else:
            raise TypeError("Invalid Comparison. VAD object can only be compared with VAD.")



    def __ge__(self, other):
        """
        Overwrites the __ge__ dunder method to compare value of two VAD objects.

                Parameters:
                        self (VAD): the VAD object that __ge__ is called upon.
                        other (VAD): the VAD object to be compared with.
    
                Returns:
                        True if self.val is greater than or equal to other.val elementwise.
                        False otherwise.
                Example:
                >>> a = VAD([2,2,3])
                >>> b = VAD([1,2,2])
                >>> a >= b
                True
        """
        if isinstance(other, VAD):
            return np.sum(self.val >= other.val) == len(self)
        else:
            raise TypeError("Invalid Comparison. VAD object can only be compared with VAD.")
            
    def isgreater(self, other):
        """
        Compare value of two VAD objects element wise

                Parameters:
                        self (VAD): the VAD object that isgreater is called upon.
                        other (VAD): the VAD object to be compared with.
    
                Returns:
                        an array of boolean values.

                Example:
                >>> a = VAD([2,2,3])
                >>> b = VAD([1,2,2])
                >>> a.isgreater(b)
                array([ True, False,  True])
        
        """
        if isinstance(other, VAD):
            return self.val > other.val      
        else:
            raise TypeError("The input must also be a VAD object.")


    def __lt__(self, other):
        """
        Overwrites the __lt__ dunder method to compare value of two VAD objects.


                Parameters:
                        self (VAD): the VAD object that __lt__ is called upon.
                        other (VAD): the VAD object to be compared with.
    
                Returns:
                        True if all values in self is less than values in other.
                        False otherwise.
                        
                Example:
                >>> a = VAD([2,2,3])
                >>> b = VAD([1,2,2])
                >>> a < b
                False
        """
        if isinstance(other, VAD):
            return np.sum(self.val < other.val) == len(self)
        else:
            raise TypeError("Invalid Comparison. VAD object can only be compared with VAD.")


    def __le__(self, other):
        """
        Overwrites the __le__ dunder method to compare value of two VAD objects.


                Parameters:
                        self (VAD): the VAD object that __le__ is called upon.
                        other (VAD): the VAD object to be compared with.
    
                Returns:
                        True if all values in self is less than or equal to values in other.
                        False otherwise.
                        
                Example:
                >>> a = VAD([2,2,3])
                >>> b = VAD([1,2,2])
                >>> a <= b
                False
        """
        if isinstance(other, VAD):
            return np.sum(self.val <= other.val) == len(self)
        else:
            raise TypeError("Invalid Comparison. VAD object can only be compared with VAD.")
            
    def isless(self, other):
        """
        Compare value of two VAD objects element wise

                Parameters:
                        self (VAD): the VAD object that isgreater is called upon.
                        other (VAD): the VAD object to be compared with.
    
                Returns:
                        an array of boolean values.

                Example:
                >>> a = VAD([2,2,3])
                >>> b = VAD([1,2,2])
                >>> a.isless(b)
                array([False, False, False])
        
        """
        if isinstance(other, VAD):
            return self.val < other.val      
        else:
            raise TypeError("The input must also be a VAD object.")
                
    ## Unary 
    def __neg__(self):
        """
        Overwrites the __neg__ dunder method to get the negation of the VAD object.
    
                Parameters:
                        self (VAD): the VAD object that __neg__ is called upon.
    
                Returns:
                        A new VAD object which has the negated value and derivative of the current VAD.

                Example:
                >>> -VAD([1,2])
                VAD(value: [-1,-2], derivatives: [[-1. -0.]
                                                  [-0. -1.]])
        """
        AD_result = self.variables * -1
        return set_VAD(AD_result) 

    ## Addition
    def __add__(self, other):
        """
        Overwrites the __add__ dunder method to apply addition to an VAD object.
    
                Parameters:
                        self (VAD): An VAD object to be applied addition to
                        other (VAD or valid input for the numpy operation): the object to be added to self
    
                Returns:
                        new_self (VAD): the new VAD object after applying addition

                Example:
                >>> a = VAD([1,2])
                >>> a + 3
                VAD(value: [4., 5.], derivatives: [[1., 0.],
                                                   [0., 1.]])
        """
        AD_result = self.variables + other
        return set_VAD(AD_result)  
        
    def __radd__(self, other):
        """
        Overwrites the __radd__ dunder method to apply addition to an VAD object 
        when the VAD object is on the right side of the addition sign.
    
                Parameters:
                        self (VAD): An AD object to be applied addition to
                        other (VAD or valid input for the numpy operation): the object to be added to self
    
                Returns:
                        new_self (VAD): the new VAD object after applying addition

                Example:
                >>> a = VAD([1,2])
                >>> 3 + a
                VAD(value: [4., 5.], derivatives: [[1., 0.],
                                                   [0., 1.]])
        """          
        return self + other
    
    def __iadd__(self, other):
        """
        Overwrites the __iadd__ dunder method to apply addition to an VAD object when the operation "+=" is used.
    
                Parameters:
                        self (VAD): An VAD object to be applied addition to
                        other (VAD or valid input for the numpy operation): the object to be added to self
    
                Returns:
                        None, but a new VAD will be assigned to the original variable.
                
                Example:
                >>> a = VAD([1,2])
                >>> a += 1
                >>> a
                VAD(value: [2., 3.], derivatives: [[1., 0.],
                                                   [0., 1.]])
        """          
        return self + other

    # Division
    def __sub__(self, other):
        """
        Overwrites the __sub__ dunder method to apply substraction to an VAD object.
    
                Parameters:
                        self (VAD): An VAD object to be applied substraction to
                        other (VAD or valid input for the numpy operation): the object to be substracted to self
    
                Returns:
                        new_self (VAD): the new VAD object after applying substraction

                Example:
                >>> a = VAD([1,2])
                >>> a - 3
                VAD(value: [-2., -1.], derivatives: [[1., 0.],
                                                     [0., 1.]])
        """  
        return self + (-1)*other
    
    def __rsub__(self, other):
        """
        Overwrites the __rsub__ dunder method to apply substraction to an VAD object.
    
                Parameters:
                        self (VAD): An VAD object to be applied substraction to
                        other (VAD or valid input for the numpy operation): the object to be substracted to self
    
                Returns:
                        new_self (VAD): the new VAD object after applying substraction

                Example:
                >>> a = VAD([1,2])
                >>> 3 - a
                VAD(value: [2., 1.], derivatives: [[-1., -0.],
                                                   [-0., -1.]])
        """  
        return (-1)*self + other
    
    def __isub__(self, other):
        """
        Overwrites the __isub__ dunder method to apply substraction to an VAD object.
    
                Parameters:
                        self (VAD): An VAD object to be applied substraction to
                        other (VAD or valid input for the numpy operation): the object to be substracted to self
    
                Returns:
                        None, but a new VAD will be assigned to the original variable.
                
                Example:
                >>> a = VAD([1,2])
                >>> a -= 3
                >>> a
                VAD(value: [-2., -1.], derivatives: [[1., 0.],
                                                     [0., 1.]])
        """        
        return self - other


    ## Mod
    def __mod__(self, other):
        """
        Overwrites the __mod__ dunder method to apply mod to an VAD object.
    
                Parameters:
                        self (VAD): An VAD object to be applied mod to
                        other (int or float): the number that self is modded by
    
                Returns:
                        new_val: the value of the VAD object after applying mod
                
                Example:
                >>> a = VAD([1,2])
                >>> a % 2
                array([1, 0])

        """        
        if isinstance(other, int) or isinstance(other, float):
            new_val = self.val % other
            print("Warning: the mod function does not yield any derivatives. Instead, the function applies mod to the current value of the AD object and returns the result.")
            return new_val
        else:
            raise TypeError("You can only mode by an integer or a float.")
    
    def __imod__(self, other):
        """
        Overwrites the __imod__ dunder method to apply mod to an VAD object when the operation "%=" is used.
    
                Parameters:
                        self (VAD): An VAD object to be applied mod to
                        other (int or float): the number that self is modded by
    
                Returns:
                        None, but an array of mod values will be assigned to the original variable.
                
                Example:
                >>> a = VAD([1,2])
                >>> a %= 2
                >>> a
                array([1, 0])

        """          
        return self % other    

    ## Multiplication
    def __mul__(self, other):
        """
        Overwrites the __mul__ dunder method to apply multiplication to an VAD object.
    
                Parameters:
                        self (VAD): An AD object to be applied multiplication to
                        other (VAD or valid input for the numpy operation): the object to be multiplied to self
    
                Returns:
                        new_self (VAD): the new VAD object after applying multiplication
                
                Example:
                >>> a = VAD([1,2])
                >>> a*2
                VAD(value: [2., 4.], derivatives: [[2., 0.],
                                                   [0., 2.]])
        """     
        AD_result = self.variables * other
        return set_VAD(AD_result)    

    def __rmul__(self, other):
        """
        Overwrites the __rmul__ dunder method to apply multiplication to an VAD object.
    
                Parameters:
                        self (VAD): An AD object to be applied multiplication to
                        other (VAD or valid input for the numpy operation): the object to be multiplied to self
    
                Returns:
                        new_self (VAD): the new VAD object after applying multiplication
                
                Example:
                >>> a = VAD([1,2])
                >>> 2*a
                VAD(value: [2., 4.], derivatives: [[2., 0.],
                                                   [0., 2.]])
        """               
        return self * other

    def __imul__(self, other):
        """
        Overwrites the __rmul__ dunder method to apply multiplication to an VAD object.
    
                Parameters:
                        self (VAD): An AD object to be applied multiplication to
                        other (VAD or valid input for the numpy operation): the object to be multiplied to self
    
                Returns:
                        None, but a new VAD will be assigned to the original variable.
                
                Example:
                >>> a = VAD([1,2])
                >>> a *= 2
                >>> a
                VAD(value: [2., 4.], derivatives: [[2., 0.],
                                                   [0., 2.]])
        """          
        return self * other
    
    
    ## Division
    def __truediv__(self, other):
        """
        Overwrites the __truediv__ dunder method to apply division to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied division to
                        other (AD or int or float): the object that self is divided by
    
                Returns:
                        new_self (AD): the new AD object after applying division
        """
        AD_result = self.variables / other
        return set_VAD(AD_result)   
        
    
    def __rtruediv__(self, other):
        """
        Overwrites the __rtruediv__ dunder method to apply division to an AD object 
        when the AD object is on the right side of the division sign.
    
                Parameters:
                        self (AD): An AD object to be applied division to
                        other (AD or int or float): the object that self is divided by
    
                Returns:
                        new_self (AD): the new AD object after applying division
        """

        AD_result = other/self.variables
        return set_VAD(AD_result)
    
    def __itruediv__(self, other):
        """
        Overwrites the __itruediv__ dunder method to apply division to an AD object when the operation "/=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied division to
                        other (AD or int or float): the object that self is divided by
    
                Returns:
                        new_self (AD): the new AD object after applying division
        """          
        return self / other
    
    
  ## power
    
    def __pow__(self, other):
        """
        Overwrites the __pow__ dunder method to apply power function to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied power function to
                        other (AD or int or float): the object that self's power will be raised to
    
                Returns:
                        new_self (AD): the new AD object after applying power function
        """           
        AD_result = self.variables ** other
        return set_VAD(AD_result)  

    
    def __ipow__(self, other):
        """
        Overwrites the __ipow__ dunder method to apply power function to an AD object when the operation "**=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied power function to
                        other (AD or int or float): the object that self's power will be raised to
    
                Returns:
                        new_self (AD): the new AD object after applying power function
        """          
        return self ** other

    
    def __rpow__(self, other):
        """
        Overwrites the __rpow__ dunder method to apply power function to an AD object 
        when the AD object is on the right side of the power sign.
    
                Parameters:
                        self (AD): An AD object to be applied power function to
                        other (AD or int or float): the object that self's power will be raised to
    
                Returns:
                        new_self (AD): the new AD object after applying power function
        """
        AD_result = other ** self.variables
        return set_VAD(AD_result)


    def diff(self, direction, order = 1):
        """
        Calculate and return the derivatives of the function represented by an AD object.
    
                Parameters:
                        self (AD): the AD object whose derivatives will be calculated.
                        direction (list): the seed indicating which variable's derivative should be returned
                        order (1 or 2): the order of the derivatives
                Returns:
                        An array representing the derivatives (or derivative) of the AD object, 
                        with directions indicted by the input direction
        """   
        if order == 1 and isinstance(direction,int):
            return self.der[:,direction]
                
        elif order == 2 and isinstance(direction, list) and len(direction) ==2:
            return self.der2[:,direction[0],direction[1]]
        else:
            raise Exception("Order exceeds 2 or length of direction and order don't match.")

# helper function
def set_VAD(ADs):
    new_val = np.concatenate([ADs[i].val for i in range(len(ADs))])
    new_der = np.array([ADs[i].der for i in range(len(ADs))])
    new_der2 = np.array([ADs[i].der2 for i in range(len(ADs))])
    return VAD(new_val, new_der, new_der2)

def my_decorator(func):
    def wrapper(vad):
        try:
            AD_result = np.array([func(ad) for ad in vad.variables])
            return set_VAD(AD_result)
        except:
            return func(vad)
    return wrapper


exp = my_decorator(admath.exp)
abs = my_decorator(admath.abs)
log = my_decorator(admath.log)
sqrt = my_decorator(admath.sqrt)
sin = my_decorator(admath.sin)
cos = my_decorator(admath.cos)
tan = my_decorator(admath.tan)
sinh = my_decorator(admath.sinh)
cosh = my_decorator(admath.cosh)
tanh = my_decorator(admath.tanh)


def pow(vad,y):
    try:
        AD_result = np.array([ad**y for ad in vad.variables])
        return set_VAD(AD_result)
    except:
        return np.power(vad,y)



# jacobian
def jacobian(funcs):
    diffs = []
    if isinstance(funcs,VAD) or isinstance(funcs,ad.AD):
        return funcs.der
    elif isinstance(funcs,list) or isinstance(funcs,np.ndarray):
        for func in funcs:
            if not isinstance(func, ad.AD):
                raise TypeError("Invalid Type. All functions should be AD object.")
            diffs.append(func.der)
        return np.vstack(diffs)

# hessian
def hessian(func):
    if isinstance(func, VAD):
        raise TypeError("Invalid Type. Sorry, we cannot handle multiple functions for Hessian.")
    elif isinstance(func, ad.AD):
        hessian = func.der2
        return hessian
    else:
        raise TypeError("Invalid Type. Function should be an AD object.")



if __name__ == "__main__":

    x = VAD([3,1])
    # f = 2*x
    g = x[1]*x[0]
    print(x)
    print(g)
    # print(f.diff(1,1))
    # print(f.diff([1,0],2))
    # print(g.diff(0,1))
    # print(g.diff(1,1))
    # print(g.diff([0,0],2))
    # #print(f)
    # print(f.diff(0,1))


    # f = admath.sin(x[0])
    # print(f)
    # g = admath.cos(x[0])
    # k = g**(-1.0)
    # print(k)
    # print(f*k)


    # f = (x[0]**3)*(x[0]**2)
    # print(f)
    # print(f.higher)
    # k = admath.sin(admath.sin(x[0]))
    # f = admath.sin(x[0])*admath.cos(x[0])
    #
    # a = admath.sin(x[0])
    # b = admath.cos(x[0])
    # # #g = admath.tan(x[0])
    # k = a*b
    # #
    # print(a)
    # print(a.higher)
    # # # h = a**(-1)
    # # # print(h)
    # # # print(h.higher)
    # print(b)
    # print(b.higher)
    # # # print (g)
    # # # print(g.higher)
    # # #mul wrong!!!!
    # print(k)
    # print(k.higher)

