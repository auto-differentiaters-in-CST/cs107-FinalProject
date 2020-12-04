import math
import numpy as np
from admath import *


class AD():

    def __init__(self, val, der=None, der2=None):
        """
        Overwrites the __init__ dunder method to create a new AD object with initial value and derivatives.
    
                Parameters:
                        val (int or float or list or np.array): the initial value of the new AD object.
                        der (int or float or list or np.array): first-order derivatives of the new AD object. 
                        der2 (int or float or list or np.array): second-order derivatives of the new AD object. 

                Returns:
                        None, but initializes an AD object when called
        """
        if isinstance(val, int) or isinstance(val, float):
            self.val = np.array(val)
        elif isinstance(val, list) or isinstance(val, np.ndarray): 
            for v in val:
                if not isinstance(val, int) and isinstance(val, float):
                    raise ValueError("Invalid input of AD object. Please initialize AD with int, float, list or array of numbers.")
            self.val = np.array(val)
        
        # Make der and der2 'private' variables so that users cannot input weird derivatives like "a"
        if der is None:
            self._der = np.array([1]*len(self))
        else:
            self._der = np.array(der)  

        if der2 is None:
            self._der2 = np.array([0]*len(self))
        else:
            self._der2 = np.array(der2) 

    def __str__(self):
        """
        Overwrites the __str__ dunder method to nicely turn an AD object into a string.
    
                Parameters:
                        self (AD): the AD object that __str__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.
        """        
        return "AD(value: {0}, first-order derivatives: {1}, second-order derivatives: {1})".format(self.val, self._der, self._der2)

    def __repr__(self):
        """
        Overwrites the __repr__ dunder method to nicely print an AD object.
    
                Parameters:
                        self (AD): the AD object that __repr__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.
        """           
        return "AD(value: {0}, first-order derivatives: {1})".format(self.val,self._der)
    
    def __len__(self):
        try:
            return len(self.val)
        except TypeError:
            return 1


    ## Unary 
    def __neg__(self):
        return 0 - self

    ## Addition
    def __add__(self, other):
        """
        Overwrites the __add__ dunder method to apply addition to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied addition to
                        other (AD or int or float): the object to be added to self
    
                Returns:
                        new_self (AD): the new AD object after applying addition
        """      
        if isinstance(other, AD):
            new_val = self.val + other.val
            new_der = np.array([self._der, other._der])
            new_der2 = np.array([self._der2, other._der2])
            return AD(val = new_val, der = new_der, der2 = new_der2)
        elif isinstance(other, int) or isinstance(other, float):
            new_val = self.val + other
            return AD(val = new_val, der = self._der, der2 = self._der2)
        else:
            raise TypeError("Invalid type. An AD object could only be added with AD or int or float.")

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
                        other (AD or int or float): the object to be added to self
    
                Returns:
                        new_self (AD): the new AD object after applying addition
        """          
        return self + other

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
                return self._der
            else:
                return self._der[direction]
        elif order == 2:
            if not direction:
                return self._der2
            else:
                return self._der2[direction]
        else:
            raise Exception("Sorry, this model can only handle first order or second order derivatives")



if __name__ == "__main__":
    x = AD([3,4])
    y = AD([1,2])
    f = x + y
    print(f)
    print(x.diff())
    print(f.diff(direction=[0,1]))

