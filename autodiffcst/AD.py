import math

class AD():

    def __init__(self, val, tags, ders=1, mode = "forward"):
        self.val = val
        if (isinstance(tags, list)) and (isinstance(ders,dict)):
            self.tags = tags
            self.ders = ders
        else:
            self.tags = [tags]
            self.ders = {tags: ders}
        self.mode = mode

    def __repr__(self):
        return "AD(value: {0}, derivatives: {1})".format(self.val,self.ders)
    
    def __str__(self):
        return "AD(value: {0}, derivatives: {1})".format(self.val,self.ders)
    
    ## Addition
    def __add__(self, other):
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
        return self + other
    
    def __iadd__(self, other):
        return self + other
    
    # Substraction
    def __sub__(self, other):
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
        return (self - other)*(-1)

    def __isub__(self, other):
        return self - other

    # Mod
    def __mod__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            new_val = self.val % other
            new_self = AD(new_val, self.tags, ders = self.ders)
            return new_self
        else:
            raise TypeError("You can only mode by an integer or a float.")

    def __imod__(self, other):
        return self % other
    
    
    ## Multiplication
    def __mul__(self, other):
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
        return self * other

    def __imul__(self, other):
        return self * other
    
    
    
    ## Division
    def __truediv__(self, other):
        try:
            self_tags = self.tags
            other_tags = other.tags

            new_ders = self.ders.copy()
            new_tags = self.tags.copy()
            
            for var in self_tags:
                if var in other_tags:
                    new_ders[var] = (other.val * self.ders[var] - self.val * other.ders[var]) / (self.val ** 2)
                else:
                    new_ders[var] = self.ders[var] / other.val
            for var in other_tags:
                if var not in self.tags:
                    new_ders[var] = -1 * other.ders[var] * self.val / (other.val**2)
                    new_tags.append(var)            

            new_val = self.val / other.val
            new_self = AD(new_val, new_tags, ders = new_ders)
            return new_self
        
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val / other
                new_ders = {}
                for var in self.tags:
                    new_ders[var] = self.ders[var] / other
                new_self = AD(new_val, self.tags, new_ders)
                return new_self
            else:
                raise TypeError("Invalid type.")

    def __rtruediv__(self, other):
        try:
            return other / self
        
        except RecursionError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = other / self.val
                new_ders = {}
                for var in self.tags:
                    new_ders[var] = self.ders[var] * -1 * other / (self.val ** 2)
                new_self = AD(new_val, self.tags, new_ders)                
                return new_self
            else:
                raise TypeError("Invalid type.")
        
    
    def __itruediv__(self, other):
        return self / other
    
    
  ## power
    def __pow__(self, other):
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
        return self ** other    

    def __rpow__(self, other):
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
        if direction == None:
            return self.ders
        try:
            return self.ders[direction]
        except KeyError:
            raise Exception("Invalid direction")



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
