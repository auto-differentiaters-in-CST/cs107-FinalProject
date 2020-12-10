import math
import numbers
import numpy as np
# import sympy as sp
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
                        tag (string or list of strings): the tag, or variable names of the new AD object, such as "x" and "y". 
                        der (float or dict): derivatives of the new AD object. 
                                              A dictionary shall be used if the object contains multiple variables.
                        order (int): the highest order of derivatives the user wants to evaluate
    
                Returns:
                        None, but initializes an AD object when called
        """
        print(higher==None)
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
                    self.higher = np.array([0] * order)
                    self.higher[0] = self.der
                    self.higher[1] = self.der2
                else:
                    print("reach here")
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
        """           
        return "AD(value: {0}, tag: {1}, derivatives: {2}, second derivatives: {3})".format(self.val,self.tag, self.der, self.der2)
    
    def __str__(self):
        """
        Overwrites the __str__ dunder method to nicely turn an AD object into a string.
    
                Parameters:
                        self (AD): the AD object that __str__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.
        """        
        return "AD(value: {0}, tag: {1}, derivatives: {2}, second derivatives: {3})".format(self.val,self.tag, self.der, self.der2)

    def __eq__(self, other):
        if isinstance(other, AD):
            if (self.val == other.val) and (self.der == other.der) and (self.der2 == other.der2): 
                return True
            else:
                return False
        else:
            raise TypeError("Invalid Comparison. AD object can only be compared with AD.")

    def __le__(self, other):
        pass
    def __ge__(self, other):
        pass
    def __len__(self):
        return len(self.tag)
    ## Unary 
    def __neg__(self):
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
    
    def __mul__(self, other):
        """
        Overwrites the __add__ dunder method to apply addition to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied addition to
                        other (AD or int or float): the object to be added to self
    
                Returns:
                        new_self (AD): the new AD object after applying addition
        """        
        try:
            new_der = self.der * other.val + self.val * other.der
            print("aaaaaaa",new_der)
            new_der2 = self.val * other.der2 + 2*other.der*self.der+other.val*self.der2
            new_val = self.val * other.val
            
            new_tag = np.unique(np.concatenate((self.tag,other.tag),0))
            # return AD(val = new_val, tag = new_tag, der = new_der, der2 = new_der2, size = self.size)
            higher_der = None
            if self.higher is not None and other.higher is not None and self.tag == other.tag :
                print("???")
                higher_der = np.array([0.0] * len(self.higher))
                higher_der[0] = new_der
                higher_der[1] = new_der2
                for i in range(0,len(self.higher)):
                    sumval = 0
                    n = i + 1
                    for k in range(n+1):
                        #print(self.higher[k-1])
                        #print(other.higher[n-k-1])
                        if k == 0:
                            sumval += choose(n,k) * self.val * other.higher[n-k-1]
                        elif k == n:
                            sumval += choose(n,k) * self.higher[k-1] * other.val
                        else:
                            sumval += choose(n,k) * self.higher[k-1] * other.higher[n-k-1]
                    higher_der[i] = sumval
            print("higher_der", higher_der)
            print("self.order", self.order)
            return AD(val=new_val,tag = new_tag, der=new_der, der2=new_der2, order = self.order, size=self.size, higher=higher_der)
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val * other
                new_der = self.der * other 
                new_der2 = self.der2 * other
                if self.higher is None:
                    #print("no higher case self,other#")
                    new_self = AD(val = new_val, tag = self.tag, der = new_der, der2 = new_der2, size = self.size)
                else:
                    #print("has higher case self,other#")
                    higher_der = other * self.higher
                    new_self = AD(val = new_val, tag = self.tag, der = new_der, der2 = new_der2, order=len(higher_der), size = self.size, higher=higher_der)
                return new_self
                    # return chain_rule(self, new_val, new_der, new_der2, higher_der = higher_der)

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
        print("call truediv")
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
        """            
        try:
            return other / self
        
        except RecursionError:
            if isinstance(other, int) or isinstance(other, float):
                return other*self **(-1)

                    #new_val = other / self.val

            #     new_der = - other / (self.val ** 2)
            #     # add second-order
            #     new_der2 = 2*other / (self.val ** 3)
            #     higher_der = None
            #     if self.higher:
            #         higher_der = np.array([0.0]*len(self.higher))
            #         #print(higher_der)
            #         higher_der[0] = new_der
            #         higher_der[1] = new_der2
            #         for i in range(2,len(self.higher)):
            #             n = i + 1
            #             coef = other*fact_ad(-1,n)
            #             mainval = math.pow(self.val[0],-n-1)
            #             higher_der[i] = coef*mainval
            #             # higher_der[i] = self.higher[i] * -1 * other / (self.higher[i-1] ** 2)
            #     return chain_rule(self, new_val, new_der, new_der2, higher_der = higher_der)
                    # return AD(val = new_val, tag = self.tag, der = new_der, der2 = new_der2, order=len(higher_der), size = self.size, higher=higher_der)
            else:
                raise TypeError("Invalid division type.")

    
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
    

    def __pow__(self, other):
        """
        Overwrites the __pow__ dunder method to apply power function to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied power function to
                        other (AD or int or float): the object that self's power will be raised to
    
                Returns:
                        new_self (AD): the new AD object after applying power function
        """
        try:
            return exp(log(self)*other)
            # new_val = self.val ** other.val
            #
            # self_der = other.val * self.val**(other.val - 1.0)
            # other_der = self.val ** other.val* np.log(self.val)
            # new_der = self_der + other_der
            #
            # x = self.val
            # y = other.val
            # h = np.array([[(y-1)*y*x**(y-2), x**(y-1)+y*x**(y-1)*np.log(x)], [x**(y-1)+y*x**(y-1)*np.log(x), x**y*(np.log(x))**2]])
            # hessian = np.zeros((self.size,self.size))
            # hessian[0:-1,0:-1] = h[:,:,0]
            # first_inner = np.array([self.der, other.der])
            # second_inner = np.zeros((self.size,self.size, self.size))
            # second_inner[0:-1,0:,0:] = np.array([self.der2, other.der2])
            # new_der2 = np.matmul(hessian, np.matmul(first_inner.T, first_inner))+np.matmul(new_der, second_inner)
            #
            # new_tag = np.unique(np.concatenate((self.tag,other.tag),0))
            #
            # # if self.higher is None:
            # return AD(val = new_val, tag = new_tag, der = new_der, der2 = new_der2, size = self.size)
    
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float) or isinstance(other, list) or isinstance(other, np.ndarray):
                try:
                    other = float(other)
                except:
                    other = np.array([float(i) for i in other])

                new_der = (self.val ** (other - 1.)) * other
                new_der2 = (self.val ** (other - 2.)) * other * (other-1.0)

                new_val = np.power(self.val, other)
        
                if self.higher is not None:
                    higher_der = np.array([1.0]*len(self.higher))
                    for i in range(len(self.higher)):
                        n = i + 1
                        
                        coef = fact_ad(other,n)
                        mainval = math.pow(self.val[0],other-n)
                        higher_der[i] = coef*mainval
                # print("here")
                # return chain_rule(self, new_val, new_der, new_der2, higher_der)
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
                # may need to change
                new_der2 = (np.log(other) ** 2) * (new_val) * self.der
                
                if self.higher is None:
                    return AD(val = new_val, tag = self.tag, der = new_der, der2 = new_der2, size = self.size)

                else:
                    higher_der = np.array([1.0]*len(self.higher))
                    for i in range(len(self.higher)):
                        n = i + 1
                        higher_der[i] = new_val * np.log(other) ** n
               
                    # return chain_rule(ad, new_val, der, der2, higher_der)
                    return AD(val = new_val, tag = self.tag, der = new_der, der2 = new_der2, order=len(higher_der), size = self.size, higher=higher_der)

                # new_self = AD(val = new_val, tag = self.tag, der = new_der, der2 = new_der2, size = self.size)
       
                # return new_self
            else:
                raise TypeError("Invalid type.") 
    

    # Differentiation
    def diff(self, direction=None, order = 1):  
        """
        Calculate and return the derivatives of the function represented by an AD object.
    
                Parameters:
                        self (AD): the AD object whose derivatives will be calculated.
                        direction (string): the seed indicating which variable's derivative should be returned
    
                Returns:
                        A dictionary (or float) representing the derivatives (or derivative) of the AD object, 
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
            raise Exception("Orders should be 1 or 2. For other orders, please use higherdiff(order).")


    # Calculate higher order derivatives
    def higherdiff(self,order):
        """
        Return the derivative of the desired order. Only works for one scalar variable and one scalar function.
    
                Parameters:
                        self (AD): the AD object whose derivatives will be calculated.
                        order (string): the order of derivative
    
                Returns:
                        the derivative of the given order evaluated at the point of self.val
        """
        if not isinstance(order, numbers.Integral):
            raise TypeError("Highest order of derivatives must be a positive integer.")
        elif order < 1:
            raise ValueError("Highest order of derivatives must be at least 1.")
        elif self.higher is None:
            raise Exception("You didn't initialize higher order")
        elif order > len(self.higher):
            #print(order)
            #print(self.order)
            raise ValueError("You asked for an order beyond what you stored.")

        return self.higher[order-1]



