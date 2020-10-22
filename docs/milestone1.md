## Section 1: Introduction
This package autodiffCST implements automatic differentiation. It can be used to automatically differentiate functions via forward mode and reverse mode, depending on the user's choice. It also provides an option of performing second order differentiation.

Differentiation, namely, the process of finding the derivatives of functions, is very prevalent in various areas of science and engineering. It can often be used to find the extrema of functions with single or multiple variables. With the advance of technology, more complicated functions and larger dataset are developed. The difficulty of performing differentiation has greatly increased and we are more dependent on computers to take derivates. Nowadays, we have three major ways of performing differentiation: symbolic, numerical and automatic (algorithmic) differentiation. We will focus on automatic differentiation for the rest of this document.

## Section 2: Background
### 2.1 An Overview of Auto Differentiation
Automatic differentiation (AD) uses algorithms to efficiently and accurately evaluating derivatives of numeric functions. It has the advantage of avoiding symbolic manipulation of functions while reaching an accuracy close to machine precision. Application of automatic differentiation includes but is not limited to astronomy, dynamic systems, numerical analysis research, optimization in finance and engineering.

The idea behind AD is to break down a function into a sequence of elementary operations and functions that have easily attained derivatives, and then sequencially apply the chain rule to evaluate the derivatives of these operations to compute the derivative of the whole function.

The two main methods of performing automatic differentiation are forward mode and reverse mode. Some other AD algorithms implement a combination of forward mode and reverse mode, but this package will implement them seperately.  

To better understand automatic differentiation, it is uncessary to get familar with some key concepts that are used in the algorithms of AD. We will use the rest of this section to briefly introduce them.

### 2.2 Elementary operations and functions
The algorithm of automatic differentiation breaks down functions into elementary arithmetic operations and elementary functions. Elementary arithmetic operations include addition, subtraction, multiplication, division and raising power (we can also consider taking roots of a number as raising it to powers less than $1$). Elementary functions include exponential, logrithmatic, and trigonometry. All of these operations and functions mentioned here have derivates that are easy to compute, so we use them as elementary steps in the evaluation trace of AD.

### 2.3 The Chain Rule
The chain rule can be used to calculate the derivate of nested functions, such in the form of $u(v(t))$. For this function, the derivative of $u$ with respect to $t$ is $$\dfrac{\partial u}{\partial t} = \dfrac{\partial u}{\partial v}\dfrac{\partial v}{\partial t}.$$

A more general form of chain rule applies when a function $h$ has several arguments, or when its argument is a vector. Suppose we have $h = h(y(t))$ where  $y \in R^n$ and $t \in R^m $. Here, $h$ is the combination of $n$ functions, each of which has $m$ variables. Using the chain rule, the derivative of $h$ with respect to $t$, now called the gradient of $h$, is

 $$    \nabla_{t}h = \sum_{i=1}^{n}{\frac{\partial h}{\partial y_{i}}\nabla y_{i}\left(t\right)}.$$

The chain rule enables us to break down complicated and nested functions into layers and operations. Our automatic differentiation algrithm sequencially sues chain rule to compute the derivative of funtions. 

### 2.4 Evaluation Trace and Computational Graph
These two concepts are the core of our automatic differentiation algorithm. Since they are so important and can be created at the same time, creating them would be the first thing to do when a function is inputted into the algorithm.

The evaluation trace tracks each layer of operations while evaluate the input function and its derivative. At each step the evaluation trace holds the traces, elementary operations, numerical values, elementary derivatives and partial derivatives. 

The computational graph is a graphical visualization of the evaluation trace. It holds the traces and elementary operations of the steps, connecting them via arrows pointing from input to output for each step. The computational graph helps us to better understand the structure of the function and its evaluation trace. Forward mode performs the operations from the start to the end of the graph or evaluation trace. Reverse mode performs the operations backwards, while applying the chain rule at each time determining the derivate of the trace.


## Section 3: How to Use AutodiffCST
Users should import the package or the class autodiffCST, 
and simply initiate the AD object by giving the function 
(or a collection of functions in a list form, representing a matrix of functions) 
they wish to differentiate. They will then be able to give a point and a direction 
and then access the derivative(s), along with some other supplementary features as 
in the code demo provided below.

```python
# import modules
import numpy as np
from AutodiffCST import AutodiffCST as adcst

# user's function, point and direction for differentiation
inputfunction = ... # should be a list of functions, can include one or more
point = ...     # the point at which we want the derivatives
direction = ...   # the direction p in our Jp

# call the class and initialize AutodiffCST object
AD = adcst(inputfunction)

"""
The following is the core.
It will get a list of values, which are the derivatives at the given point of the input functions, respectively.
- AD_type specifies the mode, the default value will be True, representing the forward mode, and False will use the reverse mode.
- tracetable gives the option of printing out a complete trace table for the computation. 
  If we say True, then we will get the table. The default setting is False as the table might be very long.
"""
AD.diff(direction, point, AD_type=True, tracetable=False)


# We could also see the graph of i-th function in our input.
# If we don't specify, then all graphs will be returned
AD.graph()[i]

"""
An option to add a new function to the input function list.
For example, suppose we originally have [xy,y-3].
We can use this function to include a third component and thus have a new function, say [xy,y-3,x-2]
"""
AD.add_function(new_function)
# Then we should do AD.diff again to obtain derivatives for the updated function matrix

# get the value of the function at the point
AD.eval(point)

# These are the most important features for our basic AD. Might add more later ...
```

## Section 4: Software Organization
The home directory of our software package would be structured as follows.

- LICENSE
- README.md
- doc/
  * quickstart_tutotial.md
  * model_documentation.md
  * testing_guidelines.md
  * concepts_explanation.md
  * references.md
- setup.py
- src/
  * \_\_init\_\_.py
  * Tree.py
  * AutodiffCST.py
  * Forward.py
  * Reverse.py
  * Extension.py

- tests/
  * test_core.py
  * test_extension.py

- TravisCI.yml
- CodeCov.yml


Specificly speaking, the README file would contain a general package description and the necessary information for users to navigate in the subdirectories. Besides, we would place our documentation, testing guidelines, a simple tutorial and relative references in the doc directory. Moreover, to package our model with PyPI, we need to include setup.py and a src directory, where stores the source code about our model. Furthermore, we would put a collection of test cases in tests directory. Last but not least, we would include TravisCI.yml and CodeCov.yml in our home directory for integrated test.

In this package, we plan to use the following public modules. 

- Modules for mathmatical calculation:
  * Numpy: we would use it for matrix operations, and basic math functions and values, such as sin, cos, \pi, e, etc. 

- Modules for testing:
  * pydoc
  * doctest 
  * Pytest

- Other modules:
  * sys
  * setuptools: we would use is for publishing our model with PyPI. 
 
To distribute our package, we would use PyPI so that users could easily install the package with *pip install*.

To better organize our software, we plan to use PyScaffold and Sphinx. The former could help us setting up the project while the latter would polish our documentation. 
 
## Section 5: Implementation

Discuss how you plan on implementing the forward mode of automatic differentiation.
- What are the core data structures?

  We would create a tree for each function to represent the graph and then follow the graph to perform calculations. 
- What classes will you implement?
  1. Forward: calculate derivatives by forward mode.
  1. Reverse: calculate derivatives by reverse mode.
  1. AutodiffCST: main class, accept input functions and return derivatives at a specific point and direction. It will also return second derivatives, graphs, and trace tables, as users need. 
  1. Tree: define basic data structure that will be used in AutodiffCST class. 
  1. Extension: apply our software to implement a useful extension or application.
 
- What method and name attributes will your classes have?
  1. Forward: 

    - Name attributes: trees representing input functions, point (at which the derivative will be calculated), direction (vector p). 
    - Method: diff(): accept all name attributes as parameters and calculate derivatives by forward mode. 
  1. Reverse: 

    - Name attributes: trees representing input functions, point (at which the derivative will be calculated), direction (vector p). 
    - Method: diff(): accept all name attributes as parameters and calculate derivatives by reverse mode. 
  1. AutodiffCST: 

    - Name attributes: input functions
    - Method: 
      - to_tree(input function): factor a input function to a series of basic operations/functions and generate a tree to represent the process.
      - diff(direction, point, AD_type, tracetable=True/False): calculate derivatives of input functions given specific direction and point. Users could indicate calculation mode by setting AD_type, and choose whether to return tracetable by setting True/False. 
      - graph(): return graph(s) of input function(s).
      - add_function(function): users can append a new function to the original list of input functions. 
  1. Tree: 
    - Name attributes: basic elements of trees. 
    - Method: get_children(), get_parent(), iterator(), etc.
  1. Extension: To be determined.
 
- What external dependencies will you rely on?

  Numpy, Math, etc. 
- How will you deal with elementary functions like sin, sqrt, log, and exp (and all the others)

  We would allow users to enter functions that can be recognized by Python, factor a input function to a series of basic operations/functions (such as sin, sqrt, log, and exp) and use if-statements to check functions and return their symbolic derivatives.
- Be sure to consider a variety of use cases. For example, don't limit your design to scalar functions of scalar values. Make sure you can handle the situations of vector functions of vectors and scalar functions of vectors. Don't forget that people will want to use your library in algorithms like Newton's method (among others).

  We allow users to enter a list of functions, which can take several variables. In addition, we take functions as input when initializing a AD object, which allow us to save the graphs in memory and calculate derivatives quickly without re-processing the functions. 

