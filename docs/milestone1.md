## Section 1: Introduction

## Section 2: Background

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

