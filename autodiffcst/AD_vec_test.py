from AD_vec import VAD
from AD import AD

if __name__ == "__main__":
    
    # 3. Vector input with a scalar function
    x = VAD([1,2,3])
    f1 = x * [2,3,4]
    f2 = (x ** 2)/2
    print(f1) # df/dx = [2,3,4]
    print("Jacobian: ", jacobian(f1))
    print("Hessian: ", hessian(f1))
    
    print(f2) # df/dx = x = [1,2,3]
    print("Jacobian: ", jacobian(f2))
    print("Hessian: ", hessian(f2))    
    
    # 3. another case: f3 should have same Jacobian as f4
    #                  f5 should have same Jacobian as f6
    x0 = AD(1,"x0")
    x1 = AD(2,"x1")
    x2 = AD(3,"x2")
    
    f3 = x[0] / x[1] - x[1] ** x[2]
    print(f3)
    print("Jacobian: ", jacobian(f3))
    
    f4 = x0 / x1 - x1 ** x2
    print(f4)
    print("Jacobian: ", jacobian(f4))  
    
    f5 = x[1] * x[1] - x[2] / x[0]
    print(f5)
    print("Jacobian: ", jacobian(f5))
    
    f6 = x1 * x1 - x2 / x0
    print(f6)
    print("Jacobian: ", jacobian(f6))     
    
    # 7. Vector input with a vector function
    f7 = [f1,f2]
    f8 = [f3,f5]
    
    print(f7) # combining df/dx from f1 and f2
    print("Jacobian: ", jacobian(f7))
    print("Hessian: ", hessian(f7))
    
    print(f8) # combining df/dx from f3 and f5
    print("Jacobian: ", jacobian(f8))
    print("Hessian: ", hessian(f8))     
    
    # optional case: not sure if we need to handle
    '''
    f9 = [f2, f3]
    print(f9) # combining df/dx from f3 and f2
    print("Jacobian: ", jacobian(f9))
    print("Hessian: ", hessian(f9))         
    '''