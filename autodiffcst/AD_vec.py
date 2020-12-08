import numpy as np
import numbers

class VAD():

    def __init__(self, val, der=None, der2=None, order = 2):
        """
        Overwrites the __init__ dunder method to create a new VAD object with initial value and derivatives.
    
                Parameters:
                        val (int or float or list or np.array): the initial value of the new VAD object.
                        der (int or float or list or np.array): first-order derivatives of the new AD object. 
                        der2 (int or float or list or np.array): second-order derivatives of the new AD object. 

                Returns:
                        None, but initializes an AD object when called
        """
        # Make der and der2 'private' variables so that users cannot input weird derivatives like "a"
    
        
        if isinstance(val, list) or isinstance(val, np.ndarray):

            for v in val:
                if not isinstance(val, numbers.Integral) and isinstance(val, float):
                    raise ValueError("Invalid input of AD object. Please initialize AD with int, float, list or array of numbers.")
            self.val = np.array(val)
        
            if der is None:
                self._der = np.ones(len(self))
                self._der2 = np.zeros(len(self))
            else:
                self._der = der
                self._der2 = der2
        else:
            raise TypeError("You need to initialize VAD with a list or a np.array. Otherwise, please use AD.")

        self.higher = None
        if isinstance(order, numbers.Integral) and order > 2:
            if len(self) == 1:
                self.higher = np.array([0] * order)
                self.higher[0] = self._der
                self.higher[1] = self._der2
            else:
                raise Exception("Cannot handle higher order derivatives for vector function")

    def __str__(self):
        """
        Overwrites the __str__ dunder method to nicely turn an AD object into a string.
    
                Parameters:
                        self (AD): the AD object that __str__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.
        """        
        return "AD(value: {0}, first-order derivatives: {1}, second-order derivatives: {2})".format(self.val, self._der, self._der2)

    def __repr__(self):
        """
        Overwrites the __repr__ dunder method to nicely print an AD object.
    
                Parameters:
                        self (AD): the AD object that __repr__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.
        """           
        return "VAD(value: {0}, first-order derivatives: {1})".format(self.val,self._der)
    
    def __len__(self):
        try:
            return len(self.val)
        except TypeError:
            return 1

    ## getter
    def __getitem__(self, pos):
        return self.val[pos]


    ## setter, Do we want this?
    def __setitem__(self, pos, val):
        self.val[pos] = val
        
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
        return self * (-1)

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
        if isinstance(other, VAD): 
            new_val = self.val + other.val
            new_der = self._der + other._der
            new_der2 = self._der2 + other._der2
            return VAD(val = new_val, der = new_der, der2 = new_der2)

        # elif isinstance(other, int) or isinstance(other, float):
        else:
            try:
                new_val = self.val + other
                return VAD(val = new_val, der = self._der, der2 = self._der2)
        
            except:
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
        if isinstance(other, VAD):
            new_val = self.val * other.val
            new_der = self._der*other.val + self.val*other._der
            new_der2 = self.val*other._der2 + 2*other._der*self._der+other.val*self._der2#self._der2 + other.val
            return VAD(val = new_val, der = new_der, der2 = new_der2)
        # elif isinstance(other, int) or isinstance(other, float):
        else:
            try:
                # other = np.array([other])
                new_val = self.val * other
                new_der = self._der * other
                
                new_der2 = self._der2 * other
                return VAD(val = new_val, der = new_der, der2 = new_der2)
            except:
                raise TypeError("Invalid type. An VAD object could only be multiply by list or array or int or float.")

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
        if isinstance(other, list):
            other = np.array(other)

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
                new_der = self._der * -1 * other / (self.val ** 2)
                # add second-order
                new_der2 = self._der2 * -1 * other / (self._der ** 2)
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
        try:
            self_der = other.val * self.val**(other.val - 1) * self._der 
            other_der = self.val ** other.val* np.log(self.val) * other._der
            new_der = self_der + other_der

            new_val = self.val ** other.val

            self_der2 = other.val * (other.val - 1)* self.val **(other.val - 2) * self._der2 
            # may need to change
            other_der2 = self._val ** other._val* np.log(self._val) * other._der2
            new_der2 = self_der2 + other_der2
            
            # add second order
            return VAD(new_val, new_der, new_der2)
            
    
        except AttributeError:
        

            if isinstance(other, int) or isinstance(other, float) or isinstance(other, list) or isinstance(other, np.ndarray):
                try:
                    other = float(other)
                except:
                    other = np.array([float(i) for i in other])

                new_val = self.val ** other
                new_der = (self.val ** (other - 1)) * other * self._der
                new_der2 = (self.val ** (other - 2)) * other * (other-1) * self._der
            
                new_self = VAD(new_val, new_der, self._der2)

                return new_self

            else:
                raise TypeError("Invalid type. Vectorized AD could only operate with int, float, list or np.ndarray. ")

    
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
                new_der = np.log(other) * (new_val) * self._der
                new_der2 = np.log(other) * (new_val) * self._der2
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
                return self._der
            else:
                
                return np.take(self._der, direction)
        elif order == 2:
            if not direction:
                return self._der2
            else:
                return self._der2[direction]
        else:
            raise Exception("Sorry, this model can only handle first order or second order derivatives")

# jacobian
def jacobian(funcs):
    diffs = []
    if isinstance(funcs,VAD):
        return funcs.diff()
    else:
        for func in funcs:
            if not isinstance(func, VAD):
                raise TypeError("All functions should be VAD object.")
            diffs.append(func.diff())
        return np.vstack(diffs)

# hessian
def hessian(func):
    
    if not isinstance(func, VAD):
        raise TypeError("All functions should be VAD object.")
    
    der2 = func.diff(order = 2)
    hessian = np.eye(len(func))*der2
    return hessian

if __name__ == "__main__":
    x = VAD([1,2,3])
    f1 = x*x*[2,3,4]
    f2 = x + [2,3,4]
    print(f1)
    print("Jacobian: ", jacobian([f1,f2]))
    print("Hessian: ", hessian(f1))