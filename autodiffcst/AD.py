import math
import numbers
import numpy as np


def _vectorize(func):
    """
    Make a function that accepts 1 or 2 arguments work with input arrays (of
    length m) in the following array length combinations:

    - m x m
    - 1 x m
    - m x 1
    - 1 x 1
    """

    def vectorized_function(*args, **kwargs):
        if len(args) == 1:
            x = args[0]
            try:
                return [vectorized_function(xi, **kwargs) for xi in x]
            except TypeError:
                return func(x, **kwargs)

        elif len(args) == 2:
            x, y = args
            try:
                return [vectorized_function(xi, yi, **kwargs)
                        for xi, yi in zip(x, y)]
            except TypeError:
                try:
                    return [vectorized_function(xi, y, **kwargs) for xi in x]
                except TypeError:
                    try:
                        return [vectorized_function(x, yi, **kwargs) for yi in y]
                    except TypeError:
                        return func(x, y, **kwargs)

    n = func.__name__
    m = func.__module__
    d = func.__doc__

    vectorized_function.__name__ = n
    vectorized_function.__module__ = m
    doc = 'Vectorized {0:} function\n'.format(n)
    if d is not None:
        doc += d
    vectorized_function.__doc__ = doc

    return vectorized_function


class AD():

    def __init__(self, val, tag=None, der=None, der2=None, size =None, order=2):
        """
        Overwrites the __init__ dunder method to create a new AD object with initial value and derivatives.
    
                Parameters:
                        val (int or float): the initial value of the new AD object.
                        tag (string or list of strings): the tag, or variable names of the new AD object, such as "x" and "y". 
                        der (float or dict): derivatives of the new AD object. 
                                              A dictionary shall be used if the object contains multiple variables.
                        mode (string: "forward" or "backward"): a string indicating the mode of the differentiation of the AD object.
    
                Returns:
                        None, but initializes an AD object when called
        """

        self.val = val if isinstance(val, np.ndarray) else np.array([val])
        if size:
            if isinstance(size, numbers.Integral):
                self.size = size 
            else:
                raise TypeError("Invalid size type. The size of AD can only be integers.")
        else:
            self.size = len(self.der)


        self.tag = tag if isinstance(tag, np.ndarray) else np.array([tag])
        
        # print(der)
        # print(type(der))
        # if der != None:
        if isinstance(der, np.ndarray):
            self.der = der
            # self.der = der if isinstance(der, np.ndarray) else np.array([der])
        else:
            self.der = np.zeros(self.size)
            self.der[tag] = 1
        
        if isinstance(der2, np.ndarray):
            self.der2 = der2
            # self.der2 = der2 if isinstance(der2, np.ndarray) else np.array([[der2]])
        else:
            self.der = np.zeros((self.size,self.size))
        
        self.order = order
        if isinstance(order, numbers.Integral) and order > 2:
            if len(self.val) == 1:
                self.higher = np.array([0] * order)
                self.higher[0] = self.der
                self.higher[1] = self.der2
            else:
                raise Exception("Cannot handle higher order derivatives for multiple variables")


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
        try:
            new_der = self.der + other.der
            new_der2 = self.der2 + other.der2
            new_val = self.val + other.val
            new_tag = np.nonzero(new_der)
            
            return AD(val = new_val, tag = new_tag, der = new_der, der2 = new_der2, size = self.size)
            
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val + other
                new_self = AD(val = new_val, tag = self.tag, der = self.der, der2 = self.der2,size = self.size)
                return new_self
            else:
                raise TypeError("Invalid type.")

    # def __add__(self, other):
    #     """
    #     Overwrites the __add__ dunder method to apply addition to an AD object.
    
    #             Parameters:
    #                     self (AD): An AD object to be applied addition to
    #                     other (AD or int or float): the object to be added to self
    
    #             Returns:
    #                     new_self (AD): the new AD object after applying addition
    #     """        
    #     # other_tag = other.tag
    #     try:
    #         new_val = self.val+other.val
    #         new_tag = self.tag
    #         new_der = self.der
    #         new_der2 = list(self.der2)

    #         for j in range(len(other.tag)):
    #             if other.tag[j] in new_tag:
    #                 loc = np.where(other.tag[j])[0]
    #                 print(np.where(other.tag[j])[0])
    #                 new_der[loc]+=other.der[j]
    #                 new_der2[loc]+=other.der2[j]
    #                 break
    #             else:
    #                 new_tag = np.append(new_tag, other.tag[j])
    #                 new_der = np.append(new_der, other.der[j])
    #                 new_der2.append(other.der2[j])
    #         result_der2 = []
    #         for i in range(len(new_der2)):
    #             result_der2.append(np.pad(new_der2[i], (0, len(new_tag) - len(new_der2[i])), mode='constant', constant_values=0))
    #         result_der2 = np.array(result_der2)
    #         if len(new_tag) == 1:
    #             new_order = max(self.order, other.order)
    #             return AD(new_val, new_tag, new_der, result_der2, new_order)
    #         else:
    #             return AD(new_val, new_tag, new_der, result_der2)
    #         return AD(new_val)
    #     except AttributeError:
    #         if isinstance(other, int) or isinstance(other, float):
    #             new_val = self.val + other
    #             new_self = AD(new_val, self.tag, der = self.der)
    #             return new_self
    #         else:
    #             raise TypeError("Invalid type.")

    # @_vectorize
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

    # @_vectorize
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
    
    # Substraction
    # @_vectorize
    # def __sub__(self, other):
    #     """
    #     Overwrites the __sub__ dunder method to apply substraction to an AD object.
    
    #             Parameters:
    #                     self (AD): An AD object to be applied substraction to
    #                     other (AD or int or float): the object to be substracted from self
    
    #             Returns:
    #                     new_self (AD): the new AD object after applying substraction
    #     """          
    #     # other_tag = other.tag
    #     try:
    #         tag1 = self.tag
    #         tag2 = other.tag

    #         new_der = self.der.copy()
    #         new_tag = self.tag.copy()
        
    #         for tag_i in tag2:
    #             if tag_i in tag1:
    #                 new_der[tag_i] -= other.der[tag_i]
    #             else:
    #                 new_der[tag_i] = -1 * other.der[tag_i]
    #                 new_tag.append(tag_i)
        
    #         new_val = self.val - other.val
    #         new_self = AD(new_val, new_tag, der = new_der)
    #         return new_self

    #     except AttributeError:
    #         if isinstance(other, int) or isinstance(other, float):
    #             new_val = self.val - other
    #             new_self = AD(new_val, self.tag, der = self.der)
    #             return new_self
    #         else:
    #             raise TypeError("Invalid type.")

    # @_vectorize
    # def __rsub__(self, other):
    #     """
    #     Overwrites the __rsub__ dunder method to apply substraction to an AD object 
    #     when the AD object is on the right side of the substraction sign.
    
    #             Parameters:
    #                     self (AD): An AD object to be applied substraction to
    #                     other (AD or int or float): the object to be substracted from self
    
    #             Returns:
    #                     new_self (AD): the new AD object after applying substraction
    #     """            
    #     return (self - other)*(-1)

    # @_vectorize
    # def __isub__(self, other):
    #     """
    #     Overwrites the __isub__ dunder method to apply substraction to an AD object when the operation "-=" is used.
    
    #             Parameters:
    #                     self (AD): An AD object to be applied substraction to
    #                     other (AD or int or float): the object to be substracted from self
    
    #             Returns:
    #                     new_self (AD): the new AD object after applying substraction
    #     """          
    #     return self - other

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

    # Mod
    # @_vectorize
    # def __mod__(self, other):
    #     """
    #     Overwrites the __mod__ dunder method to apply mod to an AD object.
    
    #             Parameters:
    #                     self (AD): An AD object to be applied mod to
    #                     other (int or float): the number that self is modded by
    
    #             Returns:
    #                     new_self (AD): the new AD object after applying mod
    #     """           
    #     if isinstance(other, int) or isinstance(other, float):
    #         new_val = self.val % other
    #         new_self = AD(new_val, self.tag, der = self.der)
    #         return new_self
    #     else:
    #         raise TypeError("You can only mode by an integer or a float.")

    # @_vectorize
    # def __imod__(self, other):
    #     """
    #     Overwrites the __imod__ dunder method to apply mod to an AD object when the operation "%=" is used.
    
    #             Parameters:
    #                     self (AD): An AD object to be applied mod to
    #                     other (int or float): the number that self is modded by
    
    #             Returns:
    #                     new_self (AD): the new AD object after applying mod
    #     """          
    #     return self % other
    
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
            new_der2 = self.val*other.der2 + 2*other.der*self.der+other.val*self.der2
            new_val = self.val * other.val
            new_tag = np.nonzero(new_der)
            
            return AD(val = new_val, tag = new_tag, der = new_der, der2 = new_der2, size = self.size)
            
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val * other
                new_der = self.der * other 
                new_der2 = self.der2 * other

                new_self = AD(val = new_val, tag = self.tag, der = new_der, der2 = new_der2, size = self.size)
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
                new_val = other / self.val
                new_der = self.der * -1 * other / (self.val ** 2)
                # add second-order
                new_der2 = self.der2 * -1 * other / (self.der ** 2)
                new_self = AD(val = new_val, tag = self.tag, der = new_der, der2 = new_der2, size = self.size)              
                return new_self
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
    # # @_vectorize
    # def __truediv__(self, other):
    #     """
    #     Overwrites the __truediv__ dunder method to apply division to an AD object.
    
    #             Parameters:
    #                     self (AD): An AD object to be applied division to
    #                     other (AD or int or float): the object that self is divided by
    
    #             Returns:
    #                     new_self (AD): the new AD object after applying division
    #     """           
    #     return self * (other ** (-1))

    # # @_vectorize
    # def __rtruediv__(self, other):
    #     """
    #     Overwrites the __rtruediv__ dunder method to apply division to an AD object 
    #     when the AD object is on the right side of the division sign.
    
    #             Parameters:
    #                     self (AD): An AD object to be applied division to
    #                     other (AD or int or float): the object that self is divided by
    
    #             Returns:
    #                     new_self (AD): the new AD object after applying division
    #     """            
    #     try:
    #         return other / self
        
    #     except RecursionError:
    #         if isinstance(other, int) or isinstance(other, float):
    #             new_val = other / self.val
    #             new_der = {}
    #             for var in self.tag:
    #                 new_der[var] = self.der[var] * -1 * other / (self.val ** 2)
    #             new_self = AD(new_val, self.tag, new_der)                
    #             return new_self
    #         else:
    #             raise TypeError("Invalid type.")

    # @_vectorize
    # def __itruediv__(self, other):
    #     """
    #     Overwrites the __itruediv__ dunder method to apply division to an AD object when the operation "/=" is used.
    
    #             Parameters:
    #                     self (AD): An AD object to be applied division to
    #                     other (AD or int or float): the object that self is divided by
    
    #             Returns:
    #                     new_self (AD): the new AD object after applying division
    #     """          
    #     return self / other
    
    
  ## power
    # @_vectorize
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
            self_der = other.val * self.val**(other.val - 1) * self.der 
            other_der = self.val ** other.val* np.log(self.val) * other.der
            new_der = self_der + other_der

            self_der2 = other.val * (other.val - 1)* self.val **(other.val - 2) * self.der2 
            # may need to change
            other_der2 = self.val ** other.val* np.log(self.val) * other.der2
            new_der2 = self_der2 + other_der2

            new_val = self.val ** other.val
            new_tag = np.nonzero(new_der)
            
            return AD(val = new_val, tag = new_tag, der = new_der, der2 = new_der2, size = self.size)
       
            # self_tag = self.tag
            # other_tag = other.tag
    
            # new_tag = self.tag.copy()
            # new_der = self.der.copy()
            
            # factor = self.val ** (other.val - 1)
            # for var in self_tag:
            #     term_1 = other.val * self.der[var]
            #     if var in other_tag:
            #         new_der[var] = factor * (term_1 + math.log(self.val) * self.val * other.der[var])
            #     else:
            #         new_der[var] = factor * term_1
            # for var in other_tag:
            #     if var not in self.tag:
            #         term_2 = math.log(self.val) * self.val * other.der[var]
            #         new_der[var] = factor * term_2
            #         new_tag.append(var)
            # new_val = self.val ** other.val
            # new_self = AD(new_val, new_tag, der = new_der)
            # return new_self
    
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float) or isinstance(other, list) or isinstance(other, np.ndarray):
                try:
                    other = float(other)
                except:
                    other = np.array([float(i) for i in other])

                new_val = self.val ** other
                new_der = (self.val ** (other - 1)) * other * self.der
                new_der2 = (self.val ** (other - 2)) * other * (other-1) * self.der
            
                new_self = AD(val = new_val, tag = new_tag, der = new_der, der2 = new_der2, size = self.size)
       
                return new_self
            # if isinstance(other, int) or isinstance(other, float):
            #     new_val = self.val ** other
            #     new_der = {}
            #     for var in self.tag:
            #         new_der[var] = (self.val ** (other - 1)) * other * self.der[var]
            #     new_self = AD(new_val, self.tag, new_der)

            #     return new_self

            else:
                raise TypeError("Invalid power type.")

    # @_vectorize
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

    # @_vectorize
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
                new_der2 = np.log(other) * (new_val) * self.der2
                
                new_self = AD(val = new_val, tag = new_tag, der = new_der, der2 = new_der2, size = self.size)
       
                return new_self
            else:
                raise TypeError("Invalid type.") 


    # Differentiation
    def diff(self, direction=None):  
        """
        Calculate and return the derivatives of the function represented by an AD object.
    
                Parameters:
                        self (AD): the AD object whose derivatives will be calculated.
                        direction (string): the seed indicating which variable's derivative should be returned
    
                Returns:
                        A dictionary (or float) representing the derivatives (or derivative) of the AD object, 
                        with directions indicted by the input direction
        """         
        if direction == None:
            return self.der
        try:
            return self.der[direction]
        except KeyError:
            raise Exception("Invalid direction")

def jacobian(ad):
    """
    Calculate and return the Jacobian of the functions represented by an AD object.

            Parameters:
                    ad (AD): the AD object whose Jacobian will be calculated.

            Returns:
                    A list representing the Jacobian of the AD object, with the order sorted by the variable names 
                    (i.e. x before y, x1 before x2, ...)
    """        
    der = ad.diff()
    der_items = list(der.items())
    der_items.sort()
    jacob = list(i[1] for i in der_items)
    return jacob






# if __name__ == "__main__":
#     x = AD(1,"x")
#     y = AD(2,"y")
#     f = 2 - x 
#     f -= 1  
#     print(f)
#     print(f.der)
#     print(f.val)
#     print(f.tag)
#     print(f.diff())