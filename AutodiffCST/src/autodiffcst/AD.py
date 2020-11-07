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
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val + other
                new_self = AD(new_val, self.tags, ders = self.ders)
            else:
                raise TypeError("Invalid type.")

    def __radd__(self, other):
        return other.add(self)
    
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
                    new_ders[tag_i] = other.ders[tag_i]
                    new_tags.append(tag_i)
        
            new_val = self.val - other.val
            new_self = AD(new_val, new_tags, ders = new_ders)
        except AttributeError:
            if isinstance(other, int) or isinstance(other, float):
                new_val = self.val - other
                new_self = AD(new_val, self.tags, ders = self.ders)
            else:
                raise TypeError("Invalid type.")


        return new_self
    
    
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

    # Differentiation
    def diff(self, direction=None):  
        if direction == None:
            return self.ders
        try:
            return self.ders[direction]
        except KeyError:
            raise Exception("Invalid direction")
    


if __name__ == "__main__":
    x = AD(1,"x")
    y = AD(2,"y")
    f = y+x
    f -= 1  
    print(f)
    print(f.ders)
    print(f.val)
    print(f.tags)
    print(f.diff())
