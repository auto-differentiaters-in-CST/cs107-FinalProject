import math

class AD():

    def __init__(self, val, tags, ders=1, mode = "forward"):
        """
        Overwrites the __init__ dunder method to create a new AD object with initial value and derivatives.
    
                Parameters:
                        val (int or float): the initial value of the new AD object.
                        tags (string or list of strings): the tags, or variable names of the new AD object, such as "x" and "y". 
                        ders (float or dict): derivatives of the new AD object. 
                                              A dictionary shall be used if the object contains multiple variables.
                        mode (string: "forward" or "backward"): a string indicating the mode of the differentiation of the AD object.
    
                Returns:
                        None, but initializes an AD object when called
        """        
        self.val = val
        if (isinstance(tags, list)) and (isinstance(ders,dict)):
            self.tags = tags
            self.ders = ders
        else:
            self.tags = [tags]
            self.ders = {tags: ders}
        self.mode = mode

    def __repr__(self):
        """
        Overwrites the __repr__ dunder method to nicely print an AD object.
    
                Parameters:
                        self (AD): the AD object that __repr__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.
        """           
        return "AD(value: {0}, derivatives: {1})".format(self.val,self.ders)
    
    def __str__(self):
        """
        Overwrites the __str__ dunder method to nicely turn an AD object into a string.
    
                Parameters:
                        self (AD): the AD object that __str__ is called upon.
    
                Returns:
                        A string containing the current value and derivatives of the AD object.
        """        
        return "AD(value: {0}, derivatives: {1})".format(self.val,self.ders)
    
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
        # other_tag = other.tags
        try:
            tags1 = self.tags
            tags2 = other.tags

            new_ders = self.ders.copy()
            new_tags = self.tags.copy()
        
            for tag_i in tags2:
                if tag_i in tags1:
                    new_ders[tag_i] += other.ders[tag_i]
                else:
                    new_ders[tag_i] = other.ders[tag_i]
                    new_tags.append(tag_i)
        
            new_val = self.val + other.val
            new_self = AD(new_val, new_tags, ders = new_ders)
            return new_self
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val + other
                new_self = AD(new_val, self.tags, ders = self.ders)
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
    
    # Substraction
    def __sub__(self, other):
        """
        Overwrites the __sub__ dunder method to apply substraction to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied substraction to
                        other (AD or int or float): the object to be substracted from self
    
                Returns:
                        new_self (AD): the new AD object after applying substraction
        """          
        # other_tag = other.tags
        try:
            tags1 = self.tags
            tags2 = other.tags

            new_ders = self.ders.copy()
            new_tags = self.tags.copy()
        
            for tag_i in tags2:
                if tag_i in tags1:
                    new_ders[tag_i] -= other.ders[tag_i]
                else:
                    new_ders[tag_i] = -1 * other.ders[tag_i]
                    new_tags.append(tag_i)
        
            new_val = self.val - other.val
            new_self = AD(new_val, new_tags, ders = new_ders)
            return new_self

        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val - other
                new_self = AD(new_val, self.tags, ders = self.ders)
                return new_self
            else:
                raise TypeError("Invalid type.")
    
    def __rsub__(self, other):
        """
        Overwrites the __rsub__ dunder method to apply substraction to an AD object 
        when the AD object is on the right side of the substraction sign.
    
                Parameters:
                        self (AD): An AD object to be applied substraction to
                        other (AD or int or float): the object to be substracted from self
    
                Returns:
                        new_self (AD): the new AD object after applying substraction
        """            
        return (self - other)*(-1)

    def __isub__(self, other):
        """
        Overwrites the __isub__ dunder method to apply substraction to an AD object when the operation "-=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied substraction to
                        other (AD or int or float): the object to be substracted from self
    
                Returns:
                        new_self (AD): the new AD object after applying substraction
        """          
        return self - other

    # Mod
    def __mod__(self, other):
        """
        Overwrites the __mod__ dunder method to apply mod to an AD object.
    
                Parameters:
                        self (AD): An AD object to be applied mod to
                        other (int or float): the number that self is modded by
    
                Returns:
                        new_self (AD): the new AD object after applying mod
        """           
        if isinstance(other, int) or isinstance(other, float):
            new_val = self.val % other
            new_self = AD(new_val, self.tags, ders = self.ders)
            return new_self
        else:
            raise TypeError("You can only mode by an integer or a float.")

    def __imod__(self, other):
        """
        Overwrites the __imod__ dunder method to apply mod to an AD object when the operation "%=" is used.
    
                Parameters:
                        self (AD): An AD object to be applied mod to
                        other (int or float): the number that self is modded by
    
                Returns:
                        new_self (AD): the new AD object after applying mod
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
        try:
            self_tags = self.tags
            other_tags = other.tags

            new_tags = self.tags.copy()
            new_ders = self.ders.copy()

            for var in self_tags:
                if var in other_tags:
                    new_ders[var] = self.ders[var] * other.val + other.ders[var] * self.val 
                else:
                    new_ders[var] = self.ders[var] * other.val
            for var in other_tags:
                if var not in self.tags:
                    new_ders[var] = other.ders[var] * self.val
                    new_tags.append(var)
            new_val = self.val * other.val
            new_self = AD(new_val, new_tags, ders = new_ders)
            return new_self

        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val * other
                new_ders = {}
                for var in self.tags:
                    new_ders[var] = self.ders[var] * other
                new_self = AD(new_val, self.tags, new_ders)
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
        return self * (other ** (-1))

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
        return other/self
        
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
            self_tags = self.tags
            other_tags = other.tags
    
            new_tags = self.tags.copy()
            new_ders = self.ders.copy()
            
            factor = self.val ** (other.val - 1)
            for var in self_tags:
                term_1 = other.val * self.ders[var]
                if var in other_tags:
                    new_ders[var] = factor * (term_1 + math.log(self.val) * self.val * other.ders[var])
                else:
                    new_ders[var] = factor * term_1
            for var in other_tags:
                if var not in self.tags:
                    term_2 = math.log(self.val) * self.val * other.ders[var]
                    new_ders[var] = factor * term_2
                    new_tags.append(var)
            new_val = self.val ** other.val
            new_self = AD(new_val, new_tags, ders = new_ders)
            return new_self
    
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val ** other
                new_ders = {}
                for var in self.tags:
                    new_ders[var] = (self.val ** (other - 1)) * other * self.ders[var]
                new_self = AD(new_val, self.tags, new_ders)

                return new_self

            else:
                raise TypeError("Invalid type.")            
    
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
                new_ders = {}
                for var in self.tags:
                    new_ders[var] = math.log(other) * (new_val) * self.ders[var]
                new_self = AD(new_val, self.tags, new_ders)  
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
            return self.ders
        try:
            return self.ders[direction]
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
    ders = ad.diff()
    ders_items = list(ders.items())
    ders_items.sort()
    jacob = list(i[1] for i in ders_items)
    return jacob

# if __name__ == "__main__":
#     x = AD(1,"x")
#     y = AD(2,"y")
#     f = 2 - x 
#     f -= 1  
#     print(f)
#     print(f.ders)
#     print(f.val)
#     print(f.tags)
#     print(f.diff())


    
    