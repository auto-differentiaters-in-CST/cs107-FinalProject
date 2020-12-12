# Project Documentation

## Section 1: Introduction
This package autodiffCST implements automatic differentiation for computational use. It can be used to automatically differentiate functions via forward mode, with the option to perform second order differentiation for inputs with a single value and a list of values and the option to perform higher order differentiation for inputs with a single value.

Differentiation, namely, the process of finding the derivatives of functions, is very prevalent in various areas of science and engineering. It can often be used to find the extrema and convexity of functions with single or multiple variables. With the advance of technology, more complicated functions and larger dataset are developed. The difficulty of performing differentiation has greatly increased and we are more dependent on computers to take derivates. Nowadays, we have three major ways of performing differentiation: symbolic, numerical and automatic (algorithmic) differentiation. We will focus on automatic differentiation for the rest of this document.

## Section 2: Background
### 2.1 An Overview of Auto Differentiation
Automatic differentiation (AD) uses algorithms to efficiently and accurately evaluate derivatives of numerical functions. It has the advantage of avoiding symbolic manipulation of functions while reaching an accuracy close to machine precision. Application of automatic differentiation includes but is not limited to astronomy, dynamic systems, numerical analysis research, optimization in finance and engineering.

The idea behind AD is to break down a function into a sequence of elementary operations and functions that have easily attained derivatives, and then sequencially apply the chain rule to evaluate the derivatives of these operations to compute the derivative of the whole function. This way, we only need to store the value of the function and its derivate at each step without the burden of parsing and remembering the whole symbolic expression. 

Forward mode and reverse mode are the two main methods to perform automatic differentiation. These two modes do not differ in their accuracy, but may differ in efficiency when the input data gets large in size. When dealing with more complexed functions or a large number of function, forward mode tends to be more efficient. Some AD algorithms even implement a combination of forward mode and reverse mode. For this project, our package only implements the forward mode and can serve as a good resource for applications such as dynamic systems and mechanical engineering. 

To better understand automatic differentiation, let's get familar with some key concepts that are used in the algorithms of AD first. We will use the rest of this section to briefly introduce them.

### 2.2 Elementary operations and functions
The algorithm of automatic differentiation breaks down functions into elementary arithmetic operations and elementary functions. Elementary arithmetic operations include addition, subtraction, multiplication, division and raising power (we can also consider taking roots of a number as raising it to powers less than $1$). Elementary functions include exponential, logrithmatic, and trigonometry. All of these operations and functions mentioned here have derivates that are easy to compute, so we use them as elementary steps in the evaluation trace of AD.

### 2.3 The Chain Rule
The chain rule can be used to calculate the derivate of nested functions, such in the form of $u(v(t))$. For this function, the derivative of $u$ with respect to $t$ is $$\dfrac{\partial u}{\partial t} = \dfrac{\partial u}{\partial v}\dfrac{\partial v}{\partial t}.$$

A more general form of chain rule applies when a function $h$ has several arguments, or when its argument is a vector. Suppose we have $h = h(y(t))$ where  $y \in R^n$ and $t \in R^m $. Here, $h$ is the combination of $n$ functions, each of which has $m$ variables. Using the chain rule, the derivative of $h$ with respect to $t$, now called the gradient of $h$, is

 $$    \nabla_{t}h = \sum_{i=1}^{n}{\frac{\partial h}{\partial y_{i}}\nabla y_{i}\left(t\right)}.$$

The chain rule enables us to break down complicated and nested functions into layers of operations. Our automatic differentiation algrithm sequencially uses chain rule to compute the derivative of funtions. 

### 2.4 Evaluation Trace and Computational Graph

The core of our automatic differentiation algorithm rests upon the idea of evaluation trace and computational graph.

The evaluation trace tracks each layer of operations while we evaluate the input function and its derivative. At each step, the evaluation trace holds the traces, elementary operations, numerical values, elementary derivatives and partial derivatives. 

The computational graph is a graphical visualization of the evaluation trace. It holds traces and elementary operations of all the steps, connecting them via arrows pointing from input to output of each step, and thus can help us better understand the structure of the function and its evaluation trace. Forward mode performs the operations from the start to the end of the graph or evaluation trace. Reverse mode performs the operations backwards, while applying the chain rule at each time determining the derivate of the trace.

Here, we provide an example of a evaluation trace and a computational graph of the function $$f(x,y)=\exp (-(\sin (x)-\cos (y))^2),$$
with derivatives evaluated at point $(\pi /2,\pi /3)$.


Evaluation trace:

|Trace|Elementary Function| &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;Current Value &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;|Elementary Function Derivative| &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp; &nbsp; &nbsp;$\nabla_x$ &nbsp; &nbsp; &nbsp; &nbsp;&nbsp;&nbsp; &nbsp; &nbsp; &nbsp;|&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp;&nbsp; &nbsp; &nbsp; &nbsp;$\nabla_y$ &nbsp; &nbsp; &nbsp; &nbsp;&nbsp; &nbsp;|
| :---:   | :-----------: | :-------:  | :-------------:       | :----------: | :-----------: |
| $x_{1}$ | $x_{1}$       | $\frac{\pi}{2}$   | $\dot{x}_{1}$           | $1$         | $0$          |
| $y_{1}$ | $y_{1}$       | $\frac{\pi}{3}$   | $\dot{y}_{1}$           | $0$         | $1$          |
| $v_{1}$ | $sin(x_{1})$  | $1$    | $cos(x_{1})\dot{x}_{1}$ | $0$         | $0$          |
| $v_{2}$ | $cos(y_{1})$  | $0.5$  | $-sin(y_{1})\dot{y}_{1}$| $0$         | $-0.866$         |
| $v_{3}$ | $v_{1}-v_{2}$ | $0.5$ | $\dot{v}_{1}-\dot{v}_{2}$| $0$        | $0.866$          |
| $v_{4}$ | $v_{3}^2$     | $0.25$ | $2v_{3}\dot{v}_{3}$     | $0$         | $0.866$          |
| $v_{5}$ | $-v_{4}$      | $-0.25$| $-\dot{v}_{4}$           | $0$         | $-0.866$         |
| $v_{6}$ | $exp(v_{5})$  | $0.779$| $exp(v_{5})\dot{v}_{5}$ | $0$         | $-0.6746$        |
| $f$     | $v_{6}$       | $0.779$| $\dot{v}_{6}$           | $0$         | $-0.6746$        |

Computational graph:

![2.4 Graph](docs/C_graph_example.jpg "Computational Graph")

In case the graph fails to show up in the notebook, please refer to docs/C_graph_example.jpg

### 2.5 Second and Higher-Order Derivatives and the Hessian Matrix

The second-order derivative of a function $f$ is the derivative of the derivative of $f$, and often referred to as the second derivative. Roughly speaking, the second derivative measures how the rate of change of a changing quantity. For example, the second derivative of the position of an object with respect to time is the instantaneous acceleration of the object, or the rate at which the velocity of the object is changing with respect to time. On the graph of a function, the second derivative corresponds to the curvature or concavity of the graph. The graph of a function with a positive second derivative is upwardly concave, while the graph of a function with a negative second derivative curves in the opposite way. Along this way, third, fourth, and higher-order derivatives can also be calculated and interpreted as the rate of change of the previous order. They can be calculated using the *Faa di Bruno Formula* -- a generalization of the chain rule as follows,

$$\frac{d^n}{dx^n}f(g(x))=\sum^n_{k=1}f^{(k)}(g(x))B_{n,k}\left(g^{(1)}(x),g^{(2)}(x),...,g^{(n-k+1)}(x)\right),$$

where the $B_{n,k}$ is the Bell polynomials

$$B_{n,k}(x_1,x_2,\ldots,x_{n-k+1})=\sum \frac{n!}{j_1!j_2!\cdots j_{n-k+1}!}\left(\frac{x_1}{1!}\right)^{j_1}\left(\frac{x_2}{2!}\right)^{j_2}\cdots \left(\frac{x_{n-k+1}}{(n-k+1)!}\right)^{j_{n-k+1}},$$
with the sum taken over all sequences $j_1,j_2,\ldots,j_{n-k+1}$ of non-negative integers such that these two conditions are satisfied:

- $j_1+j_2+j_3+\cdots+j_{n-k+1}=k$,
- $j_1+2j_2+3j_3+\cdots+(n-k+1)j_{n-k+1}=n.$

Similarly, we have a generalization of the product rule, the *Leibniz Rule*,
$$\frac{d^n}{dx^n}\left(f(x)g(x)\right)=\sum^n_{k=1}\begin{pmatrix} n\\k \end{pmatrix} f^{(k)}(x)g^{(n-k)}(x).$$

The Hessian matrix or Hessian is a square matrix of second-order partial derivatives of a scalar-valued function, or scalar field. It describes the local curvature of a function of many variables. We will use the Hessian matrix to report the second-order derivatives of the functions.

Recall that for a scalar-valued function of multiple scalar variables, say $f(x_1,x_2,\ldots,x_n)$, the Jacobian matrix summarize the first-order derivatives with respect to each variable as follows:
\begin{align*}
{J}(f)= \begin{bmatrix}
    \frac{\partial f}{\partial x_1}, &
     \frac{\partial f}{\partial x_2},&
    \cdots,&
    \frac{\partial f}{\partial x_n}\\
  \end{bmatrix}. 
\end{align*}
And then we observe that Hessian matrix looks like 
\begin{align*}
{H}(f)= \begin{bmatrix}
    \frac{\partial^2f}{\partial x^2_1} &\frac{\partial^2f}{\partial x_1\partial x_2} &\cdots &\frac{\partial^2f}{\partial x_1\partial x_n} \\
      \frac{\partial^2f}{\partial x_2\partial x_1} &\frac{\partial^2f}{\partial x^2_2} &\cdots &\frac{\partial^2f}{\partial x_2\partial x_n} \\
    \vdots  &\vdots &\ddots &\vdots \\
    \frac{\partial^2f}{\partial x_n\partial x_1} &\frac{\partial^2f}{\partial x_n\partial x_2} &\cdots &\frac{\partial^2f}{\partial x^2_n} \\
  \end{bmatrix}. 
\end{align*}

With more complicated case, when $f = \begin{bmatrix}f_1(x_1,x_2,\ldots,x_n),&f_2(x_1,x_2,\ldots,x_n),&\cdots,&f_m(x_1,x_2,\ldots,x_n)\end{bmatrix}$, we will now have the Jacobian
\begin{align*}
{J}(f)= \begin{bmatrix}
    \frac{\partial f_1}{\partial x_1} &
     \frac{\partial f_1}{\partial x_2}&
    \cdots &
    \frac{\partial f_1}{\partial x_n}\\
\frac{\partial f_2}{\partial x_1} &
     \frac{\partial f_2}{\partial x_2}&
    \cdots&
    \frac{\partial f_2}{\partial x_n}\\
    \vdots  &\vdots &\ddots &\vdots \\
    \frac{\partial f_m}{\partial x_1} &
     \frac{\partial f_m}{\partial x_2}&
    \cdots &
    \frac{\partial f_m}{\partial x_n}\\
  \end{bmatrix}, 
\end{align*}
and the Hessian will be a tensor in $3D$ according to the rules.

## Section 3: How to Use autodiffCST


**Installation**

Our package is for Python 3 only. To install autodiffCST, you need to have pip3 installed first. If you don't, please install pip3 following these instructions https://pip.pypa.io/en/stable/installing/.

Then, you could install this package by running 
```pip3 install autodiffCST``` from the command line. 

*Please be aware that you might need to manually install sympy and numpy prior to installing our package, depending on your environment. You could do so by running ```pip3 install sympy``` and ```pip3 install numpy``` in your terminal.*

An alternative is to clone our repository by running ```git clone https://github.com/auto-differentiaters-in-CST/cs107-FinalProject.git``` from the command line and then ```cd autodiffcst``` to go to the directory where the modules reside. Then use ```pip install -r requirements.txt``` to install the required pacakges.

**User Guide**

After installation, you could import this package by ```import autodiffcst as cst```.

Then, you could initiate the VAD object by giving the point where you wish to differentiate. VAD can take in a vector input values, representing a point's coordinates in multi-dimensional space. Moreover, you could also try other supplementary features as in the code demo provided below. Notice that to use this demo, you can find a dem.py file in the `autodiffcst` directory.

A brief guide on the main features before code examples:

- autodiffcst.**VAD** *(val,order=2)*: 

    Initialize a list of scalar variables with respect to which you wish to evaluate the derivatives.
    When you want to do differentiation to a function of multiple scalar variables, you must build all of them at once within the same **VAD** call.
    
    Can also initialize as a single vector variable, but restrictions may apply. See example 3.3.

    - Parameters: 
        - *val*: a list or a numpy array of numbers. The value of your variable.
        - *order*: a positive integer. The highest order of derivative to be evaluated. Can only be set as greater than 2 if *val* has a single number.
    
    - Returns:    None


- autodiffcst.**diff** *(direction, order = 1)*:

    A method to get derivative in a specified direction once variable(s) or a function made up of established variable(s) is set up. 

    - Parameters: 
        - *direction*: an index number if order = 1, a list of two index numbers if order = 2. The index of the variable at which you want to obtain the derivative.
        - *order*: either 1 or 2. The order of the derivative you want. Should only be 1 or 2.
        
    When you want to get a first derivative, you must specify one variable; when you want to get a second derivative, you must specify two variables.
    Even if you only have one variable, you still need to specify 0 for first derivative and $[0,0]$ for second derivative.

    - Returns:  The value of the desired derivative.
        


- autodiffcst.**jacobian** *(f)*:

    A function to get first derivatives of all variables in the form of a Jacobian matrix.

    - Parameters:
        - *f*: a single function or a list or numpy array of function, each made up of variables built with **VAD**.
    
    - Returns: The Jacobian matrix.


- autodiffcst.**hessian** *(f)*:

    A function to get second derivatives of all variables in the form of a Hessian matrix. Can only be used when applicable.

    - Parameters:
        - *f*: a single function made up of only scalar variables built with **VAD**. Can't be used with vector variables.

    - Returns: The Hessian matrix.
        

- autodiffcst.**higherdiff** *(order)*:
    
    A method to get higher-order derivatives. Only work for the case of single variables. In order to use this feature, you should initialize your variable from **VAD** with input *val* being a list of a single value, and then specify the highest *order*.

    - Parameters:
        - *order*: a positive number.
    
    - Returns: The derivative as a number to the specified order.


### Example 3.1
Simple case: a list of a single scalar variable. First-order, second-order, and higher-order derivatives can be calculated.

```python
# import modules
>>> import autodiffcst as cst

>>> [u] = cst.VAD([5])           # initialize VAD objects u with a single point at 5
# u = cst.VAD([5]) can also be used, but then you need to refer to your scalar variable as u[0].

>>> f = u*2-3                    # build a function with VAD object

>>> print(f)                     # print f's value and derivative
AD(value: [7], derivatives: [2.])
    
>>> dfdu = f.diff(0)             # get derivative in the direction of u
>>> print(dfdu)                  # notice that you must specify a direction
2.0

>>> dfdu2 = f.diff([0,0],order=2)# get second derivative df^2/dudu
>>> print(dfdu2)
0.0

>>> dfdu = f.diff()              # get derivative without direction, get an error
TypeError: diff() missing 1 required positional argument: 'direction'

>>> cst.jacobian(f)              # Jacobian matrix for single variable and single function, if you wish
array([[2.]])

>>> cst.hessian(f)               # Hessian matrix for single variable and single function, if you wish
array([[0.]])

# only in this simple case, we support higher-order derivatives beyond 2
>>> [x] = cst.VAD([2],order=10)  # initialize as before, but specify that you want to get to order up tp 10

>>> g = 2*cst.exp(x)
>>> g.higherdiff(10)
14.7781121978613

>>> g.higherdiff(12)             # since you initialize with order=10, you cannot go beyond that
ValueError: You asked for an order beyond what you stored.

>>> f = x**3                  # let's try another case for higher-order derivatives
>>> f.higher 
array([12., 12.,  6.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
# Hurray! Now you get all first ten derivatives at once!

```

### Example 3.2
Advanced cases: initialize VAD objects with vectors (multiple input values)

```python
# import modules
>>> import autodiffcst as cst

>>> x, y, z = cst.VAD([1,2,3])     # initialize VAD objects x, y, z with values 1, 2, 3 respectively
                                   # with multiple variable, you can skip brackets

>>> f1,f2,f3= x+y, x**2+z, x*y*z   # build three functions with x, y, z
>>> print(f3)                      # print f3's values and derivatives
AD(value: [6], tag: [0 1 2], derivatives: [6. 3. 2.])  

>>> cst.jacobian([f1, f2, f3])   
array([[1., 1., 0.],
       [2., 0., 1.],
       [6., 3., 2.]])

>>> cst.jacobian(f1)                # you can also get jacobian for only f1
array([1., 1., 0.])

>>> cst.hessian(f1)
array([[0., 0., 0.],
       [0., 0., 0.],
       [0., 0., 0.]])

>>> cst.hessian(f3)
array([[0., 3., 2.],
       [3., 0., 1.],
       [2., 1., 0.]])

>>> cst.hessian([f1, f2, f3])     # the hessian of this object will be a tensor, and is not supported
TypeError: Invalid Type. Function should be an AD object.

>>> f1.diff(0,order=1)            # get first derivative with respect to x (index 0)
1.0

>>> f3.diff([2,1],order=2)        # get second derivative with respect to z and y
1.0

>>> f3.diff([1,2],order=2)        # notice that hessian matrix is symmetric, so we get the same value
1.0

>>> x, y, z = cst.VAD([1,2,3],order=3)  # higher-order feature is no longer supported for multiple scalar variables
Exception: We cannot handle derivatives of order > 2 for more than one scalar variables.
```

### Example 3.3
Tricky case: using **VAD** to create a vector variable

For example, suppose we have $x=[1,2,3]$, and we wish to see the derivative of $f(x)=\sin(x)$. Here this $f$ is, in fact, an abuse of notation, because what it really says is $$f = \begin{bmatrix} f_1(x), f_2(x), f_3(x)\end{bmatrix} = \begin{bmatrix} \sin (x_1), \sin (x_2), \sin (x_3)\end{bmatrix}.$$
But our package is capable of handling this case, as long as:
- $f$ is only one function, rather than a list of multiple functions.
- You have only one such vector $x$ created as one VAD object.

Our ```hessian``` only handles cases of up to $2D$ matrices and so you will not be able to see the second derivative tensor. But we do store the hessian correctly in our object, as you can see in our demo below.

Please be informed that we DO NOT support a single function applied to multiple vector variables, such as $f = e^x+3y$ where $x=[1,2,3]$ and $y=[4,5,6]$. And this case is exempted from implementation as specified in piazza @595.

However, if you decide to be a crooked user regardless and try to do this with our package, our algorithm will take $x,y$ as the same vector variable $[x_1,x_2,x_3]$ and return results based on that assumption. Having said that, if you call ```jacobian```, the result will be computed as in the case where $f = e^x+3x$, and therefore is NOT the correct solution. Of course, you won't be able to use ```hessian``` method at all in this case. 
We just want you to be aware that there will be NO error or exception raised when you create $f = e^x+3y$, as this is beyond the scope of this project. But this doesn't mean we can handle it properly for you. Therefore, it's your responsibility to take extra caution when reaching this tricky case and avoid the abusive use of our package. 

```python
# import modules
>>> import autodiffcst as cst

>>> v = cst.VAD([1,2,3])        # initialize VAD objects: a vector v of value [1,2,3]
    
>>> f = cst.sin(v)              # build VAD: a single function applied to the vector v
>>> print(f)                    # print f's value and derivative. Here the second derivative will appear as a 3x3 matrix 
VAD(value: [[0.84147098]
            [0.90929743]
            [0.14112001]], derivatives: [[ 0.54030231  0.          0.        ]
                                         [-0.         -0.41614684 -0.        ]
                                         [-0.         -0.         -0.9899925 ]])   

>>> cst.jacobian(f)
array([[ 0.54030231,  0.        ,  0.        ],
       [-0.        , -0.41614684, -0.        ],
       [-0.        , -0.        , -0.9899925 ]])

>>> cst.hessian(f)              # you will get an error, as promised.
TypeError: Invalid Type. Sorry, we cannot handle multiple functions for Hessian.

>>> f.diff(0,order=1)           # get the first derivative with respect to v[0] (or x0), the first variable         
array([ 0.54030231, -0.        , -0.        ])
# the result is the first column of jacobian, since we want [df1/dx0,df2/dx0,df3/dx0]

>>> f.diff([2,2],order=2)       # now the second derivative df^2/dx3dx3 is also a vector 
array([ 0.        , -0.        , -0.14112001])

# now check the bonus!
>>> f.der2                      # only in this case, you will be able to get the tensor hessian
array([[[-0.84147098,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ],
        [ 0.        ,  0.        ,  0.        ]],

       [[-0.        , -0.        , -0.        ],
        [-0.        , -0.90929743, -0.        ],
        [-0.        , -0.        , -0.        ]],

       [[-0.        , -0.        , -0.        ],
        [-0.        , -0.        , -0.        ],
        [-0.        , -0.        , -0.14112001]]])
```

### Example 3.4 
Using autodiffcst in real case: finding minimum of Rosenbrock function with Newton's method

The Rosenbrock function (https://en.wikipedia.org/wiki/Rosenbrock_function):

$$ f(x,y)=100(y-x^2)^2+(1-x)^2,$$

is a common test function used for optimization algorithms. We use it to demonstrate a nice application of our package.
Here we only show the code for doing Newton's method in each iteration, where our package can be used to obtain the Jacobian and Hessian directly. Please refer to docs/using_VAD_for_Newtons_method.ipynb for the complete notebook with a contour plot of the optimization path.
Please refer to docs/using_VAD_for_Newtons_method.ipynb for the complete notebook.


```python
import numpy as np
import autodiffcst as cst

# using VAD to create variables at point (x_i,y_i)
[a,b] = cst.VAD([x_i,y_i])
# construct the function
Rsbrk = 100*(b-a**2)**2+(1-a)**2
# Take a Newton step by solving the linear system 
# constructed using the Hessian and gradient
step = np.linalg.solve(cst.hessian(Rsbrk),-cst.jacobian(Rsbrk))
x_i += step[0]
y_i += step[1]
```


## Section 4: Software Organization
The home directory of our software package would be structured as follows.

- LICENSE.txt
- README.md
- requirements.txt
- docs/
    * README.md
    * milestone1.ipynb
    * milestone2.ipynb
    * milestone2_progress.ipynb
    * documentation.ipynb
    * documentation.md
    * api
    * using_VAD_for_Newtons_method.ipynb
 
- setup.py
- demo.ipynb
- src/
    - autodiffcst/
        * \_\_init\_\_.py
        * AD.py
        * AD_vec.py
        * admath.py

- tests/
    * AD_test.py
    * test_admath.py

- TravisCI.yml
- CodeCov.yml

Specificly speaking, the README file would contain a general package description and the necessary information for users to navigate in the subdirectories. Besides, we would place our documentation, testing api, and previous milestone files in the `docs` directory. Moreover, to package our model with PyPI, we need to include `setup.py` and an `autodiffcst` directory in `src` directory, where stores the source code of our package. These core modulesinclude: `AD.py` (used by `AD_vec` to handle single value inputs), `AD_vec.py` (defines the main object class VAD and some of its related functions), and `admath.py` (defines elementary math operations for VAD objects). Furthermore, we would put a collection of test cases in `tests` directory. The tests are run through Pytest. Last but not least, we would include TravisCI.yml and CodeCov.yml in our home directory for integrated test. In addition, we also included a simple tutorial `demo.ipynb` for demo of our code in the home directory.

To distribute our package, we would use PyPI so that users could easily install the package with *pip install autodiffCST*.
For developers, the repository can be cloned by running git clone https://github.com/auto-differentiaters-in-CST/cs107-FinalProject.git from the command line.

## Section 5: Implementation Details

Our core classes are the VAD object class and AD object class. Notice that the user of our package will only directly interact with VAD objects. AD objects are used by the VAD class to handle single value inputs and perform basic operations for each element in vector inputs. 

In the VAD class, the main data structure we used is Numpy array. Most of the core attributes of the VAD object: value (`val`), first (`der`), second (`der2`) and higher (`higher`) order derivatives are all stored using Numpy array. As shown by the \_\_init\_\_ function of VAD below, the other attribute of VAD is `order`, which is an integer indicating the derivative order to which we wish to calculate. All of the derivatives up to this order will be stored in the attribute `higher`. Notice that since our package only handles higher derivatives for single value inputs, an `order` input greater than 2 is only valid when the `val` input is a list containing one single value. 

The AD class has two more attributes than the VAD class: `size` and `tag`. `size` is an integer representing the total number of inputs, or the dimension of the whole VAD object that the AD object is a part of. `tag` is a list of integers representing the direction of the AD object in its hosting VAD object (for example, $[1,0]$). 

Notice that except for `val`, all the other attributes to initialize VAD are not required and not recommended for users to specify. They will be handles by the module automatically, as do all the attributes in the AD class. A slight exception is `order`. We need to specify it when doing higher order (>2) derivative for single value input.

Some of the major methods for VAD is listed below (this is not an exhaustive list and only for informative purpose):
- a constructor
``` python
def __init__(self, val, der=None, der2=None, order=2, higher=None):
```  
- overloaded dunder methods as follows:
``` python
__add__
__sub__
__pow__
__mul__
__mod__
__div__
``` 
&ensp; and more basic operations according to https://www.python-course.eu/python3_magic_methods.php

- a `diff` method, which takes in a direction as a vector and an integer order, and returns the corresponding derivative of the function. This function only handles vector input values and derivatives of order 2 and 1. Higher order derivatives for single value input are handled separately as shown in the demo above.

- a `jacobian` method, which takes in a vector of VAD functions and returns the Jacobian matrix.

- a `hessian` method, which takes in a single VAD function and returns the Hessian matrix.

The module `admath.py` contains other elementary functions that are used to form functions using VAD, so it needs to be imported along side with `AD_vec.py`. The functions in `admath.py` include `exp`, `abs`, `log`, `sqrt`, `sin`, `cos`, `tan`, `sinh`, `cosh`, and `tanh`.

In this package, we will use the following public modules to deal with elementary functions, we would allow users to enter functions that can be recognized by Python, factor a input function to a series of basic operations/functions (such as sin, sqrt, log, and exp), as in Section 3: How to Use.

- Modules for mathmatical calculation:
  * Numpy: we use it for matrix operations, and basic math functions and values, such as sin, cos, $\pi$, e, etc. 
  * Sympy: we use it to compute the Bell polynomials for higher order derivatives.

- Modules for testing:
  * pydoc
  * doctest 
  * Pytest

- Other modules:
  * sys
  * setuptools: we would use is for publishing our model with PyPI. 

# Extension

Here is the future features section from *Milestone 2*:

1. Differentiate a list of functions. Our package can deal with one function with multiple varaibles. In the future we plan to take a list of functions as input and output its Jacobian accordingly. Using Numpy array as the data structure to keep the Jacobian would be ideal, so we will need to change the implementation of our current jacobian method. 

2. Backward Mode. Right now our mode for doing automatic differetiation is defaulted to forward mode, because we have not implemented backward mode yet. We would need new functions that use the AD object class to implement backward mode. To keep track of the traces, we need to create a trace table, possibly using Numpy array, in the function that runs backward mode. 

3. Newton's method. We would like to use our AD package to solve meaningful problems. One way to achieve this is to use it in an implementation of Newton's method. This will be a script that imports our AD package to calculate the derivatives in Newton's method.

After some consideration, we decided that we would like to do ***higher order derivatives*** instead of Backward Mode and Newton's method as proposed. This pivot shifting is approved, so we updated the future features section to reflect this change and M2 feedback (the update was made before the whole module was finished so some function names and implementations might not match):
#### 1. Differentiate a list of functions. 
Our package can deal with one function with multiple varaibles. In the future we plan to take a list of functions as input and output its Jacobian accordingly. Using Numpy array as the data structure to keep the Jacobian would be ideal, so we will need to change the implementation of our current jacobian method.

#### 2. Higher order derivatives. 
A starting point would be allowing second order derivatives taken on our AD objects and returning the correct Jacobian matrix accordingly. Note that this cannot be achieved by simply applying `diff()` to an AD object twices, since the Jacobian matrix would be different and the datatype would be different. We would need to store the values of the second derivatives of our AD object at each elementary steps in the evaluation trace. Then we would need another function to return the second derivatives (possibly named `second_diff()`), which functions similarly to `diff()`, but returns the second derivatives of the AD object. Apart from the `jacobian()` function, we will also have a `hessian()` function which returns the second order derivatives matrix of the function. 


### Description of Extension and its Background
As mentioned above, our main extension is calculating higher order derivatives using automatic differentiation. Higher order derivatives are prevalent in Numerical Analysis researches, mechanical engineering, astronomy and a number of other fields of application. Most of the times, people are interesting in using derivatives of order 1, 2 and 3, seldomly 4 and 5. Orders higher than such are rarely consider except for pure mathematical and academic purposes. More mathematical background of higher order derivatives is provided in the Background section above. 

Our implementation of higher order derivatives is integrated with our main object classes `VAD`. The features differ for single value input and vector input of `VAD`, so we will introduce them separately. 

For `VAD` and functions of `VAD` with single value input, we can calculate their derivatives up to an arbitrary order specified by the attribute `order`. These high order derivatives are stored in an attribute `higher` and can be accessed through it. Notice that due to the nature of differentiation, most derivatives will become 0 after some times of differentiations, except for the iterative functions including `sin` and `cos`. 

For `VAD` and functions of `VAD` with vector value input, we can calculate their derivatives up to the second order. This shall be enough for basic applications of differentiating vectors. The first and second order derivatves are stored in attributes `der` and `der2` respectively and can be accessed through them and through functions `jacobian()` and `hessian()`.

In addition, we demoed a case where our extension could aid Newton's method in the file `using_VAD_for_Newtons_method.ipynb` in the `docs` directory. It is quite an integrated test scheme which utilizes a large coverage of the elementary operations that can be used in our modules. It is highly encouraged to try this test case. 

# Broader Impact and Inclusivity Statement

### Broader Impact
We hope our package would be applied to different fields that require doing differentiations via computer programs: physics, engineering, applied mathematics, astronomy, and even other areas that the developers of this package have never imagined. We hope this package can be used to do automatic differentiations accurately and efficiently and can inspire the development of enhanced versions of automatic differentiation packages in the future. We see a number of possibilities that this package could be enhanced and would be happy to see them completed. 

On the other hand, we do not hope to see that this package is used for plagiarism, cheating, or shortcut for doing differentiation. The open-source nature of this package makes it accessible to people, but also susceptible to people who plan to use it for plagiarism. Users should be aware of this nature and wisely choose their way of using this package. This package is not designed for shortcuts of doing differentiation practices. People could use it to check their answers for calculating derivatives by hand or by other algorithms, but should not use it in place of derivative calculation practices. These practices have their purposes and using this package to get the answers does not contribute to the learning process.

We also see that when working on this project, we connected mathematical ideas such as Leibniz Rule and Faa di Bruno Formula to our automatic differentiation algorithms. Although this should not be the first time when people used these formulas to calculate higher-order derivatives, it was inspiring for us to do the implementation ourselves. We hope our project serves as a case where we bridge the gap between theories and applications. This experience will allow us and many students alike to keep striving for this goal and further tells that this is the best time when all kinds of knowledge come together to facilitate new discoveries.

### Software Inclusivity
The autodiffCST package and its developer welcome users who are contributors from all backgrounds and identities. We believe excellence in a collaborative project comes from trust, respect, and caring for each other, as it is evident through the process of developing this package. We tried our best to make our package as inclusive and user-friendly as possible with the willingness to reach more people that are interested in this package, by providing fitting documentation and instructions. Admittedly, this package is written in English and Python, but we welcome the contribution from people that are fluent in any language and programming languages. During the process of developing this package, pull requests are reviewed and approved by all developers. Whenever one of us feels the need to initiate a pull request, this person would communicate with other members and reach an agreement together. We would love to bring this positive communication to a future collaboration of this package and beyond.

# Future

1. In the future, we would like our package to be able to implement higher (>2) order derivatives for vector inputs. We now can handle up to second order, but did not get to implement orders higher than that. Such orders could be useful for some applications.
2. We would like to handle functions that contains multiple vector inputs. As of now, we can only perform operations with one vector input `VAD` object, or multiple single value input `VAD` objects. If we were to implement functions with nultiple vector inputs, the Jacobian and Hessian matrices would be much more complicated than what we have now. This could serve as a fitting extension for furture implementations.
3. Improve computation efficiency. For instance, we used the Faa di Bruno Formula to calculate higher order derivatives, but is there a more efficient approach, in terms of both time and storage complexity? Possible candidates are using symbolic expression of the function, using Backward Mode and using other formulas. It is yet to explore which one is the most efficient option.
4. Further applications. As mentioned above, our modules can already be used in Newton's method and fits applications in areas such as mechanical engineering and dynamic system. As of the higher order derivatve extension, it can be useful in Numerical Analysis and pedagogical purposes. Physics is another area of possible application of our package, since second order derivatives are prevalent.
