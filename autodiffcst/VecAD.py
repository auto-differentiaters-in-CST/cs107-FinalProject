from autodiffcst import AD as ad


class VecAD:
    def __init__(self, ads):
        self.ads = ads

    def jacobian(self):
        [ad.jacobian(a) for a in self.ads]




