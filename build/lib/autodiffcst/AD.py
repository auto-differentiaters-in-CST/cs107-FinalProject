import math
import numbers
import numpy as np
import warnings
import os
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from autodiffcst.admath import chain_rule,fact_ad,choose, log, exp
# from admath import chain_rule,fact_ad,choose

class AD():

    def __init__(self, val, order=2, size = None, tag=None, der=None, der2=None,  higher=None): 
        """
        Overwrites the __init__ dunder method to create a new AD object with initial value and derivatives.
    
                Parameters:
                        val (int or float): the initial value of the new AD object.
                        order (int): the highest order of derivatives the user wants to evaluate
                        size (int): the size of dimension that the AD object resides
                        tag (list of int or np.ndarray of int): the tag, or direction that the AD object resides in its dimension
                        der (list of float or np.ndarray of float): first order derivative of the new AD object, 
                                                                    contained in a list or or np.ndarray           
                        der2 (list of float or np.ndarray of float): second order derivative of the new AD object, 
                                                                    contained in a list or or np.ndarray
                        higher (list of float or np.ndarray of float): higher order derivatives of the new AD object, 
                                                                    contained in a list or or np.ndarray

                Returns:
                        None, but initializes an AD object when called

                Example:
                >>> AD(val=1, order=2, size=1, tag=1) 
                AD(value: [1], derivatives: [1.])

                >>> AD(1, 2, 1, 1) 
                AD(value: [1], derivatives: [1.])

                Note: Initializing val, order, size and tag is a must to use AD directly
        """
        self.val = val if isinstance(val, np.ndarray) else np.array([val])
        if der is None:
            if size is None:
                self.size = 1
                warnings.warn("Size is not specified. The default size is set to 1.", Warning)
            elif isinstance(size, numbers.Integral):
                self.size = size 
            else:
                raise TypeError("Invalid size type. The size of AD can only be integers.")
        else:
            self.size = len(der)


        self.tag = tag if isinstance(tag, np.ndarray) else np.array([tag])
        
        if isinstance(der, np.ndarray):
            self.der = der
        else:
            self.der = np.zeros(self.size)
            self.der[tag] = 1
        
        if isinstance(der2, np.ndarray):
            self.der2 = der2
        else:
            self.der2 = np.zeros((self.size,self.size))
        
        self.order = order
        # self.higher= higher
        if isinstance(order, numbers.Integral):
            if len(self.val) == 1 and order > 2:
                if higher is None:
                    self.higher = np.array([0.0] * order)
                    self.higher[0] = self.der
                    self.higher[1] = self.der2
                else:
                    #print("reach here")
                    self.higher = higher
            elif order > 2:
                raise Exception("Cannot handle higher order derivatives for multiple variables")    
            elif order > 0:
                # must always have higher attribute
                self.higher = None
            else:
                raise ValueError("Order of derivative must be at least 1.")
                
        else:
            raise TypeError("Invalid input for order of derivative.")


    def __repr__(self):
        """
        Overwrites the __repr__ dunder method to nicely print an AD object.
    
                Parameters:
                        self (AD): the AD object that __repr__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.

                Example:
                >>> repr(AD(1, 2, 1, 0))
                "AD(value: [1], derivatives: [1.])"
        """
        return "AD(value: {0}, derivatives: {1})".format(self.val, self.der)

    def __str__(self):
        """
        Overwrites the __str__ dunder method to nicely turn an AD object into a string.
    
                Parameters:
                        self (AD): the AD object that __str__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.

                Example:
                >>> str(AD(1, 2, 1, 0))
                "AD(value: [1], derivatives: [1.])"
        """        
        return "AD(value: {0}, derivatives: {1})".format(self.val,self.der)

    def __eq__(self, other):
        """
        Overwrites the __eq__ dunder method to check if two AD objects are equal in value.
    
                Parameters:
                        self (AD): the AD object that __eq__ is called upon.
                        other (AD): the AD object to be compared with.
    
                Returns:
                        True if the two AD objects have equal values.
                        False if the two AD objects do not have equal values.

                Example:
                >>> AD(1, 2, 1, 0) == AD(2, 2 ,1, 0)
                False
        """  
        if isinstance(other, AD):
            if self.val == other.val: 
                return True
            else:
                return False
        else:
            raise TypeError("Invalid Comparison. AD object can only be compared with AD.")
    
    def __ne__(self, other):
        """
        Overwrites the __ne__ dunder method to check if two AD objects are not equal in value.
    
                Parameters:
                        self (AD): the AD object that __ne__ is called upon.
                        other (AD): the AD object to be compared with.
    
                Returns:
                        False if the two AD objects have equal values.
                        True if the two AD objects do not have equal values.

                Example:
                >>> AD(1, 2, 1, 0) != AD(2, 2 , 1, 0)
                True
        """ 
        return not self == other
    
    def fullequal(self, other):
        """
        Check if two AD objects are equal in value, first and second derivatives.
    
                Parameters:
                        self (AD): the AD object that fullequal is called upon.
                        other (AD): the AD object to be compared with.
    
                Returns:
                        True if the two AD objects have equal values, first and second derivatives.
                        False if the two AD objects differ in any of their equal values, first and second derivatives.

                Example:
                >>> a = AD(1, 2, 1, 0)
                >>> a.fullequal(AD(2, 2 ,1, 0))
                False
        """ 
        if isinstance(other, AD):
            return np.allclose(self.val, other.val) and np.allclose(self.der,other.der) and np.allclose(self.der2, other.der2)
        else:
            raise TypeError("Invalid Comparison. AD object can only be compared with AD.")


    def __lt__(self, other):
        """
        Overwrites the __lt__ dunder method to check if the current AD objects is less than another AD in value.
    
                Parameters:
                        self (AD): the AD object that __lt__ is called upon.
                        other (AD): the AD object to be compared with.
    
                Returns:
                        True if the current AD objects is less than the other AD in value.
                        False if the current AD objects is not less than the other AD in value.

                Example:
                >>> AD(1, 2, 1, 0) < AD(2, 2 ,1, 0)
                array([ True])
        """  
        if isinstance(other, AD):
            return self.val < other.val
        else:
            raise TypeError("Invalid Comparison. AD object can only be compared with AD.")

    def __gt__(self, other):
        """
        Overwrites the __gt__ dunder method to check if the current AD objects is greater than another AD in value.
    
                Parameters:
                        self (AD): the AD object that __gt__ is called upon.
                        other (AD): the AD object to be compared with.
    
                Returns:
                        True if the current AD objects is greater than the other AD in value.
                        False if the current AD objects is not greater than the other AD in value.

                Example:
                >>> AD(1, 2, 1, 0) > AD(2, 2 ,1, 0)
                array([False])
        """ 
        if isinstance(other, AD):
            return self.val > other.val
        else:
            raise TypeError("Invalid Comparison. AD object can only be compared with AD.")

    def __le__(self, other):
        """
        Overwrites the __le__ dunder method to check if the current AD objects is no larger than another AD in value.
    
                Parameters:
                        self (AD): the AD object that __le__ is called upon.
                        other (AD): the AD object to be compared with.
    
                Returns:
                        True if the current AD objects is no larger than the other AD in value.
                        False if the current AD objects is larger than the other AD in value.

                Example:
                >>> AD(1, 2, 1, 0) <= AD(2, 2 ,1, 0)
                array([ True])
        """ 
        if isinstance(other, AD):
            return self.val <= other.val
        else:
            raise TypeError("Invalid Comparison. AD object can only be compared with AD.")


    def __ge__(self, other):
        """
        Overwrites the __ge__ dunder method to check if the current AD objects is no less than another AD in value.
    
                Parameters:
                        self (AD): the AD object that __ge__ is called upon.
                        other (AD): the AD object to be compared with.
    
                Returns:
                        True if the current AD objects is no less than the other AD in value.
                        False if the current AD objects is less than the other AD in value.

                Example:
                >>> AD(1, 2, 1, 0) >= AD(2, 2 ,1, 0)
                array([False])
        """  
        if isinstance(other, AD):
            return self.val >= other.val
        else:
            raise TypeError("Invalid Comparison. AD object can only be compared with AD.")


    def __len__(self):
        """
        Overwrites the __len__ dunder method to get the length of the vector dimension that the AD object resides in.
    
                Parameters:
                        self (AD): the AD object that __len__ is called upon.
    
                Returns:
                        An integer representing the length of the vector dimension that the AD object resides in.

                Example:
                >>> len(AD(1, 2, 1, 0))
                1
        """ 
        return len(self.tag)
    ## Unary 
    def __neg__(self):
        """
        Overwrites the __neg__ dunder method to get the negation of the AD object.
    
                Parameters:
                        self (AD): the AD object that __neg__ is called upon.
    
                Returns:
                        A new AD object which has the negated value and derivative of the current AD.

                Example:
                >>> -AD(1, 2, 1, 0)
                AD(value: [-1], derivatives: [-1.])
        """
        return self*(-1)
    ## Addition
    def __add__(self, other):
        """
        Overwrites the __add__ dunder method to apply addition to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied addition to
                        other (AD or int or float): the object to be added to self
    
                Returns:
                        new_self (AD): the new AD object after applying addition
                
                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> y = AD(1, 2, 2, 1)
                >>> x + y
                AD(value: [4], derivatives: [1., 1.])
        """        
        try:
            new_der = self.der + other.der
            new_der2 = self.der2 + other.der2
            new_val = self.val + other.val

            new_tag = np.unique(np.concatenate((self.tag,other.tag),0))#np.nonzero(new_der)
            if self.higher is None or other.higher is None:
                return AD(val = new_val, tag = new_tag, der = new_der, der2 = new_der2, size = self.size)
            else:
                if len(self.higher) != len(other.higher):
                    raise Exception("The two object are not initialized with the same highest order.")
                
                return AD(val = new_val, tag = new_tag, der = new_der, der2 = new_der2, order=len(self.higher),size = self.size,higher=self.higher+other.higher)
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):

                new_val = self.val + other
                if self.higher is None:
                    new_self = AD(val=new_val, tag=self.tag, der=self.der, der2=self.der2, size=self.size)
                else:
                    new_self = AD(val = new_val, tag = self.tag, der = self.der, der2 = self.der2,order=len(self.higher),size = self.size,higher=self.higher)
                return new_self
            else:
                raise TypeError("Invalid type.")

    def __radd__(self, other):
        """
        Overwrites the __radd__ dunder method to apply addition to an AD object 
        when the AD object is on the right side of the addition sign.
    
                Parameters:
                        self (AD): An AD object to be applied addition to
                        other (AD or int or float): the object to be added to self
    
                Returns:
                        new_self (AD): the new AD object after applying addition
                
                Example:
                >>> x = AD(3, 2, 1, 0)
                >>> 2 + x
                AD(value: [5.], derivatives: [1.])
        """          
        return self + other

    def __iadd__(self, other):
        """
        Overwrites the __iadd__ dunder method to apply addition to an AD object when the operation "+=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied addition to
                        other (AD or int or float): the object to be added to self
    
                Returns:
                        None, but a new AD will be assigned to the original variable.
                
                Example:
                >>> x = AD(3, 2, 1, 0)
                >>> x += 2
                >>> x
                AD(value: [5.], derivatives: [1.])
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
                
                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> y = AD(1, 2, 2, 1)
                >>> x - y
                AD(value: [2.], derivatives: [1.，-1.])
        """      
        return self + (-1)*other
    
    def __rsub__(self, other):
        """
        Overwrites the __rsub__ dunder method to apply substraction to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied substraction to
                        other (AD or valid input for the numpy operation): the object to be substracted from self
    
                Returns:
                        new_self (AD): the new AD object after applying substraction
                
                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> 2-x
                AD(value: [-1.], derivatives: [-1.，-0.])
        """      
        
        return (-1)*self + other
    
    def __isub__(self, other):
        """
        Overwrites the __isub__ dunder method to apply substraction to an AD object when the operation "-=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied substraction to
                        other (AD or valid input for the numpy operation): the object to be substracted from self
    
                Returns:
                        None, but a new AD will be assigned to the original variable.

                Example:
                >>> x = AD(3, 2, 1, 0)
                >>> x -= 2
                >>> x 
                AD(value: [1.], derivatives: [1.])
        """          
        return self - other
    
    def __mul__(self, other):
        """
        Overwrites the __mul__ dunder method to apply multiplication to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied multiplication to
                        other (AD or int or float): the object to be added to self
    
                Returns:
                        new_self (AD): the new AD object after applying multiplication

                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> y = AD(1, 2, 2, 1)
                >>> x * y
                AD(value: [3.], derivatives: [1.，3.])
        """        
        try:
            
            new_der = self.der * other.val + self.val * other.der
            
            new_der2 = self.val * other.der2 + np.matmul(np.array([other.der]).T,np.array([self.der]))  \
                        + np.matmul(np.array([self.der]).T,np.array([other.der]))+ other.val * self.der2

            new_val = self.val * other.val
            
            new_tag = np.unique(np.concatenate((self.tag,other.tag),0))
            # return AD(val = new_val, tag = new_tag, der = new_der, der2 = new_der2, size = self.size)
            higher_der = None
            if self.higher is not None and other.higher is not None and self.tag == other.tag :
                #print("???")
                higher_der = np.array([0.0] * len(self.higher))
                higher_der[0] = new_der
                higher_der[1] = new_der2
                for i in range(0,len(self.higher)):
                    sumval = 0
                    n = i + 1
                    for k in range(n+1):
                        if k == 0:
                            sumval += choose(n,k) * self.val * other.higher[n-k-1]
                        elif k == n:
                            sumval += choose(n,k) * self.higher[k-1] * other.val
                        else:
                            sumval += choose(n,k) * self.higher[k-1] * other.higher[n-k-1]
                    higher_der[i] = sumval
            
            return AD(val=new_val,tag = new_tag, der=new_der, der2=new_der2, order = self.order, size=self.size, higher=higher_der)
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val * other
                new_der = self.der * other 
                new_der2 = self.der2 * other
                if self.higher is None:
                    new_self = AD(val = new_val, tag = self.tag, der = new_der, der2 = new_der2, size = self.size)
                else:
                    higher_der = other * self.higher
                    new_self = AD(val = new_val, tag = self.tag, der = new_der, der2 = new_der2, order=len(higher_der), size = self.size, higher=higher_der)
                return new_self
                    
            else:
                raise TypeError("Invalid type.")

    def __rmul__(self, other):
        """
        Overwrites the __rmul__ dunder method to apply multiplication to an AD object 
        when the AD object is on the right side of the multiplication sign.
    
                Parameters:
                        self (AD): An AD object to be applied multiplication to
                        other (AD or int or float): the object to be multiplied to self
    
                Returns:
                        new_self (AD): the new AD object after applying multiplication
                
                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> 2*x 
                AD(value: [6.], derivatives: [2.，0.])
        """            
        return self * other

    def __imul__(self, other):
        """
        Overwrites the __imul__ dunder method to apply multiplication to an AD object when the operation "*=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied multiplication to
                        other (AD or int or float): the object to be multiplied to self
    
                Returns:
                        None, but a new AD will be assigned to the original variable.
                
                Example:
                >>> x = AD(3, 2, 1, 0)
                >>> x *= 5
                >>> x
                AD(value: [15.], derivatives: [5.])
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

                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> y = AD(1, 2, 2, 1)
                >>> x / y
                AD(value: [3.], derivatives: [1.，-2.])
        """
        return self * (other ** (-1.0))

    
    def __rtruediv__(self, other):
        """
        Overwrites the __rtruediv__ dunder method to apply division to an AD object 
        when the AD object is on the right side of the division sign.
    
                Parameters:
                        self (AD): An AD object to be applied division to
                        other (AD or int or float): the object that self is divided by
    
                Returns:
                        new_self (AD): the new AD object after applying division
                
                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> 3/x 
                AD(value: [1.], derivatives: [-0.33333333, -0.])
        """            
        try:
            return other / self
        
        except RecursionError:
            if isinstance(other, int) or isinstance(other, float):
                return other*self **(-1)            
            else:
                raise TypeError("Invalid division type.")

    
    def __itruediv__(self, other):
        """
        Overwrites the __itruediv__ dunder method to apply division to an AD object when the operation "/=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied division to
                        other (AD or int or float): the object that self is divided by
    
                Returns:
                        None, but a new AD will be assigned to the original variable.
                
                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> x /= 3
                >>> x 
                AD(value: [1.], derivatives: [0.33333333, 0.])
        """          
        return self / other
    

    def __pow__(self, other):
        """
        Overwrites the __pow__ dunder method to apply power function to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied power function to
                        other (AD or int or float): the object that self's power will be raised to
    
                Returns:
                        new_self (AD): the new AD object after applying power function
                
                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> y = AD(1, 2, 2, 1)
                >>> x ** y
                AD(value: [3.], derivatives: [1., 3.29583687])
        """
        if isinstance(other, AD):
            if self.val[0] != 0:
                return exp(log(self)*other)
            else:
                if other.val <= 2:
                    raise ValueError("Derivative is undefined.")
                else:
                    return AD(0, tag = self.tag, der = np.zeros(self.size), der2 = np.zeros((self.size, self.size)))
    
        elif isinstance(other, int) or isinstance(other, float) or isinstance(other, list) or isinstance(other, np.ndarray):
            try:
                other = float(other)
            except:
                other = np.array([float(i) for i in other])

            new_der = (self.val ** (other - 1.)) * other
            new_der2 = (self.val ** (other - 2.)) * other * (other-1.0)

            new_val = np.power(self.val, other)
            higher_der = None
            if self.higher is not None:
                higher_der = np.array([1.0]*len(self.higher))
                for i in range(len(self.higher)):
                    n = i + 1

                    coef = fact_ad(other,n)
                    mainval = math.pow(self.val[0],other-n)
                    higher_der[i] = coef*mainval
            
            return chain_rule(self, new_val, new_der, new_der2, higher_der = higher_der)

        else:
            raise TypeError("Invalid power type.")

    def __ipow__(self, other):
        """
        Overwrites the __ipow__ dunder method to apply power function to an AD object when the operation "**=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied power function to
                        other (AD or int or float): the object that self's power will be raised to
    
                Returns:
                        new_self (AD): the new AD object after applying power function
                
                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> 3 ** x
                AD(value: [27.], derivatives: [29.66253179, 0.])
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
                        None, but a new AD will be assigned to the original variable.
                
                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> x **= 2
                >>> x
                AD(value: [9.], derivatives: [6., 0.])
        """            
        try:
            return other ** self
        
        except RecursionError:
            if isinstance(other, int) or isinstance(other, float):
                return exp(log(other) * self)
            else:
                raise TypeError("Invalid type.") 
    

    # Differentiation
    def diff(self, direction=None, order = 1):  
        """
        Calculate and return the derivatives of the function represented by an AD object.
    
                Parameters:
                        self (AD): the AD object whose derivatives will be calculated.
                        direction (int): the seed indicating which variable's derivative should be returned
    
                Returns:
                        derivative (float): the derivative of the AD object, wrt the input direction 
                
                Example:
                >>> x = AD(3, 2, 2, 0)
                >>> y = AD(1, 2, 2, 1)
                >>> f = x / y
                >>> dfdy = f.diff(1)
                >>> dfdy 
                -3.0
                
        """
        if order == 1 and isinstance(direction, int):
            return self.der[direction]

        elif order == 2 and isinstance(direction, list) and len(direction) == 2:
            return self.der2[direction[0], direction[1]]
        else:
            raise Exception("Order exceeds 2 or length of direction and order don't match.")


    # Calculate higher order derivatives
    def higherdiff(self,order):
        """
        Return the derivative of the desired order. Only works for one scalar variable and one scalar function.
    
                Parameters:
                        self (AD): the AD object whose derivatives will be calculated.
                        order (string): the order of derivative
    
                Returns:
                        the derivative(float) of the given order evaluated at the point of self.val
                
                Example:
                >>> x = AD(val = 3, order = 10, size = 1, tag = 0)
                >>> f = x**5
                >>> f.higherdiff(5)
                120.0
                >>> f.higherdiff(6)
                0.0

        """
        if not isinstance(order, numbers.Integral):
            raise TypeError("Highest order of derivatives must be a positive integer.")
        elif order < 1:
            raise ValueError("Highest order of derivatives must be at least 1.")
        elif self.higher is None:
            raise Exception("You didn't initialize higher order")
        elif order > len(self.higher):
            raise ValueError("You asked for an order beyond what you stored.")

        return self.higher[order-1]



