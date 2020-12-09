import numpy as np
import numbers
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

import autodiffcst.AD as ad
import autodiffcst.admath as admath

class VAD():

    def __init__(self, val, der=None, der2=None):
        """
        Overwrites the __init__ dunder method to create a new VAD object with initial value and derivatives.
    
                Parameters:
                        val (int or float or list or np.array): the initial value of the new VAD object.
                        der (int or float or list or np.array): first-order derivatives of the new AD object. 
                        der2 (int or float or list or np.array): second-order derivatives of the new AD object. 

                Returns:
                        None, but initializes an AD object when called
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
        
        arr_ad = np.array([None]*len(self))
        for i in range(len(self)):
            arr_ad[i] = ad.AD(val=self.val[i], tag=self.tag[i], size = self.size,
                            der=self.der[i], der2=self.der2[i])
        self.variables = arr_ad

    def __str__(self):
        """
        Overwrites the __str__ dunder method to nicely turn an AD object into a string.
    
                Parameters:
                        self (AD): the AD object that __str__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.
        """
        return "VAD(value: {0}, tag: {1}, derivatives: {2}, second derivatives: {3})".format(self.val, self.tag,
                                                                                             self.der, self.der2)

    def __repr__(self):
        """
        Overwrites the __repr__ dunder method to nicely print an AD object.
    
                Parameters:
                        self (AD): the AD object that __repr__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.
        """
        return "VAD(value: {0}, tag: {1}, derivatives: {2}, second derivatives: {3})".format(self.val, self.tag,
                                                                                             self.der, self.der2)

    def __len__(self):
        try:
            return len(self.val)
        except TypeError:
            return 1

    ## getter
    def __getitem__(self, pos):
        return self.variables[pos]


    ## setter, Do we want this?
    def __setitem__(self, pos, newAD):
        self.variables[pos] = newAD
        
    # Comparison Equal
    def __eq__(self, other):
        """
        compare value of two VAD objects 
        for example:
            >>> a = AD([1,2,3])
            >>> b = AD([2,2,3])
            >>> a == b
            >>> False
        """
        if isinstance(other, VAD):
            if np.sum(self.val == other.val) == len(self): 
                return True
                
            else:
                return False
        else:
            raise TypeError("Invalid Comparison. VAD object can only be compared with VAD.")
            
    def isequal(self, other):
        """
        compare value of two VAD objects element wise
        for example:
            a = VAD([1,2,3])
            b = VAD([2,2,3])
            >>> a.isequal(b)
            array([False,True,True]) 
        """
        if isinstance(other, VAD):
            return self.val == other.val      
        else:
            raise TypeError("The input must also be a VAD object.")

    def __ge__(self, other):
        """
        compare value of two VAD objects 
        for example:
            >>> a = AD([1,2,3])
            >>> b = AD([2,2,3])
            >>> a > b
            >>> False
        """
        if isinstance(other, VAD):
            if np.sum(self.val > other.val) == len(self): 
                return True
                
            else:
                return False
        else:
            raise TypeError("Invalid Comparison. VAD object can only be compared with VAD.")
            
    def isgreater(self, other):
        """
        compare value of two VAD objects element wise
        for example:
            a = VAD([1,2,3])
            b = VAD([2,2,3])
            >>> a.isgreater(b)
            array([False,True,True]) 
        """
        if isinstance(other, VAD):
            return self.val > other.val      
        else:
            raise TypeError("The input must also be a VAD object.")
    
    def __le__(self, other):
        """
        compare value of two VAD objects 
        for example:
            >>> a = AD([1,2,3])
            >>> b = AD([2,2,3])
            >>> a < b
            >>> False
        """
        if isinstance(other, VAD):
            if np.sum(self.val < other.val) == len(self): 
                return True
                
            else:
                return False
        else:
            raise TypeError("Invalid Comparison. VAD object can only be compared with VAD.")
            
    def isless(self, other):
        """
        compare value of two VAD objects element wise
        for example:
            a = VAD([1,2,3])
            b = VAD([2,2,3])
            >>> a.isless(b)
            array([True,False,False]) 
        """
        if isinstance(other, VAD):
            return self.val < other.val      
        else:
            raise TypeError("The input must also be a VAD object.")
                
    ## Unary 
    def __neg__(self):
        AD_result = self.variables * -1
        return set_VAD(AD_result) 

    ## Addition
    def __add__(self, other):
        """
        Overwrites the __add__ dunder method to apply addition to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied addition to
                        other (AD or valid input for the numpy operation): the object to be added to self
    
                Returns:
                        new_self (AD): the new AD object after applying addition
        """
        AD_result = self.variables + other
        return set_VAD(AD_result)  
        
    def __radd__(self, other):
        """
        Overwrites the __radd__ dunder method to apply addition to an AD object 
        when the AD object is on the right side of the addition sign.
    
                Parameters:
                        self (AD): An AD object to be applied addition to
                        other (AD or int or float): the object to be added to self
    
                Returns:
                        new_self (AD): the new AD object after applying addition
        """          
        return self + other
    
    def __iadd__(self, other):
        """
        Overwrites the __iadd__ dunder method to apply addition to an AD object when the operation "+=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied addition to
                        other (AD or valid input for the numpy operation): the object to be added to self
    
                Returns:
                        new_self (AD): the new AD object after applying addition
        """          
        return self + other

    # Division
    def __sub__(self, other):
        """
        Overwrites the __sub__ dunder method to apply substraction to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied substraction to
                        other (AD or valid input for the numpy operation): the object to be substracted from self
    
                Returns:
                        new_self (AD): the new AD object after applying substraction
        """      
        return self + (-1)*other
    
    def __rsub__(self, other):
        return (-1)*self + other
    
    def __isub__(self, other):
        """
        Overwrites the __isub__ dunder method to apply substraction to an AD object when the operation "-=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied substraction to
                        other (AD or valid input for the numpy operation): the object to be substracted from self
    
                Returns:
                        new_self (AD): the new AD object after applying substraction
        """          
        return self - other


    ## Mod
    def __mod__(self, other):
        """
        Overwrites the __mod__ dunder method to apply mod to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied mod to
                        other (int or float): the number that self is modded by
    
                Returns:
                        new_val: the value of the AD object after applying mod
        """        
        if isinstance(other, int) or isinstance(other, float):
            new_val = self.val % other
            print("Warning: the mod function does not yield any derivatives. Instead, the function applies mod to the current value of the AD object and returns the result.")
            return new_val
        else:
            raise TypeError("You can only mode by an integer or a float.")
    
    def __imod__(self, other):
        """
        Overwrites the __imod__ dunder method to apply mod to an AD object when the operation "%=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied mod to
                        other (int or float): the number that self is modded by
    
                Returns:
                        new_val: the value of the AD object after applying mod
        """          
        return self % other    

    ## Multiplication
    def __mul__(self, other):
        """
        Overwrites the __mul__ dunder method to apply multiplication to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied multiplication to
                        other (AD or int or float): the object to be multiplied to self
    
                Returns:
                        new_self (AD): the new AD object after applying multiplication
        """     
        AD_result = self.variables * other
        return set_VAD(AD_result)    

    def __rmul__(self, other):
        """
        Overwrites the __rmul__ dunder method to apply multiplication to an AD object 
        when the AD object is on the right side of the multiplication sign.
    
                Parameters:
                        self (AD): An AD object to be applied multiplication to
                        other (AD or int or float): the object to be multiplied to self
    
                Returns:
                        new_self (AD): the new AD object after applying multiplication
        """            
        return self * other

    def __imul__(self, other):
        """
        Overwrites the __imul__ dunder method to apply multiplication to an AD object when the operation "*=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied multiplication to
                        other (AD or int or float): the object to be multiplied to self
    
                Returns:
                        new_self (AD): the new AD object after applying multiplication
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
        
        try:
            return other / self
        
        except RecursionError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = other / self.val
                new_der = self.der * -1 * other / (self.val ** 2)
                # add second-order
                new_der2 = self.der2 * -1 * other / (self.der ** 2)
                new_self = VAD(new_val, new_der, new_der2)                
                return new_self
            else:
                raise TypeError("Invalid type.")

    
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
        try:
            return other ** self
        
        except RecursionError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = other ** self.val
                new_der = np.log(other) * (new_val) * self.der
                new_der2 = np.log(other) * (new_val) * self.der2
                new_self = VAD(new_val, new_der, new_der2)  
                return new_self
            else:
                raise TypeError("Invalid type.") 

    
    def diff(self, direction=None, order = 1):
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
        if order == 1:
            if not direction:
                return self.der
            else:
                
                return np.take(self.der, direction)
        elif order == 2:
            if not direction:
                return self.der2
            else:
                return self.der2[direction]
        else:
            raise Exception("Sorry, this model can only handle first order or second order derivatives")

# helper function
def set_VAD(ADs):
    new_val = np.array([ADs[i].val for i in range(len(ADs))])
    new_der = np.array([ADs[i].der for i in range(len(ADs))])
    new_der2 = np.array([ADs[i].der2 for i in range(len(ADs))])
    return VAD(new_val, new_der, new_der2)

# jacobian
def jacobian(funcs):
    diffs = []
    if isinstance(funcs,VAD) or isinstance(funcs,ad.AD):
        return funcs.diff()
    elif isinstance(funcs,list) or isinstance(funcs,np.ndarray):
        for func in funcs:
            if not isinstance(func, ad.AD):
                raise TypeError("Invalid Type. All functions should be AD object.")
            diffs.append(func.diff())
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


def sin(vad):
    AD_result = np.array([admath.sin(ad) for ad in vad.variables])
    return set_VAD(AD_result)

def cos(vad):
    AD_result = np.array([admath.cos(ad) for ad in vad.variables])
    return set_VAD(AD_result)


if __name__ == "__main__":
    x = VAD([1,4,3])
    [a,b,c] = VAD([1,2,3]) 
    print(a.der)
    '''
    print("case1")
    y = x * 3
    print(y.diff())
    '''
    print("----------------")
    print("case2")
<<<<<<< HEAD
    
    f = 2**x.variables[1]
    print(f)

    '''
    print(jacobian([f]))
    print(hessian(f))
    print("----------------")
    print("case3")
    g = x/3
    print(g)
    print("--------------)
    print("case4")
    h = f - x.variables[2]
    print(h)
    '''
