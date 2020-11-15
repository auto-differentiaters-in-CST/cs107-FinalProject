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
            print(new_val, new_tags, new_ders)
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
    # super not sure how tags work here
    def __div__(self, other):
        try:
            tags1 = self.tags
            tags2 = other.tags

            new_ders = self.ders.copy()
            new_tags = self.tags.copy()

            for tag_i in tags2:
                if tag_i in tags1:
                    new_ders[tag_i] = (other.val * self.ders[tag_i] - self.val * other.ders[tag_i]) / (self.val ** 2)
                else:
                    new_ders[tag_i] = -1* self.val / (other.ders[tag_i]**2)
                    new_tags.append(tag_i)

            new_val = self.val / other.val
            new_self = AD(new_val, new_tags, ders = new_ders)
            return new_self
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val / other
                new_ders = self.ders / other
                new_self = AD(new_val, self.tags, new_ders)
            else:
                raise TypeError("Invalid type.")

    def __rdiv__(self, other):
        return other.div(self)
    
    def __idiv__(self, other):
        return self / other
    
    
    ## power
    # adapted from ryt's pow()
    # more general, can be used by **
    # do we need x ** y, if so, what's the derivative
    def __pow__(self, other):
        if isinstance(other, int) or isinstance(other, float):
            new_val = math.pow(self.val, other)
            der = other * math.pow(self.val, other - 1)
            new_self = AD(new_val, self.tags, der)
            return new_self
        else:
            raise TypeError("You can only power by an integer or a float.")        
    
    def __ipow__(self, other):
        return self ** other    


    # Differentiation
    def diff(self, direction=None):  
        if direction == None:
            return self.ders
        try:
            return self.ders[direction]
        except KeyError:
            raise Exception("Invalid direction")

    
def chain_rule(ad, new_val, der):
    new_ders = dict()
    for tag in ad.tags:
        new_ders[tag] = der * ad.ders[tag]
    new_ad = AD(new_val, ad.tags, new_ders)
    return new_ad

def abs(ad):
    new_val = math.fabs(ad.val)
    if ad.val > 0:
        der = 1
    elif ad.val < 0:
        der = -1
    else:
        raise Exception("Derivative undefined")
    return chain_rule(ad, new_val, der)


def exp(ad):
    new_val = math.exp(ad.val)
    der = new_val
    return chain_rule(ad, new_val, der)


def log(ad): #consider different base?
    new_val = math.log(ad.val)
    der = 1/ad.val
    return chain_rule(ad, new_val, der)

def pow(ad, y):
    new_val = math.pow(ad.val, y)
    der = y * math.pow(ad.val, y - 1)
    return chain_rule(ad, new_val, der)

def sqrt(ad):
    return ad ** 0.5

if __name__ == "__main__":
    x = AD(1,"x")
    y = AD(2,"y")
    f = 2 - x 
    f -= 1  
    print(f)
    print(f.ders)
    print(f.val)
    print(f.tags)
    print(f.diff())
