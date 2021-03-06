# cs107-FinalProject
## This is the repository for CS107 group project for fall 2020 at Harvard IACS.
### Group 5
Contributors: Xiaohan Yang, Max Li, Runting Yang,Hanwen Zhang

travis badge:
[![Build Status](https://travis-ci.com/auto-differentiaters-in-CST/cs107-FinalProject.svg?token=AjVcVSqkqdiJgwaimWYR&branch=master)](https://travis-ci.com/auto-differentiaters-in-CST/cs107-FinalProject)

codecov badge:
[![codecov](https://codecov.io/gh/auto-differentiaters-in-CST/cs107-FinalProject/branch/master/graph/badge.svg?token=US1Y8Z9OE0)](https://codecov.io/gh/auto-differentiaters-in-CST/cs107-FinalProject)


## Introduction
This package autodiffCST implements automatic differentiation for computational use. It can be used to automatically differentiate functions via forward mode, with the option to perform second order differentiation for inputs with a single value and a list of values and the option to perform higher order differentiation for inputs with a single value.

Differentiation, namely, the process of finding the derivatives of functions, is very prevalent in various areas of science and engineering. It can often be used to find the extrema and convexity of functions with single or multiple variables. With the advance of technology, more complicated functions and larger dataset are developed. The difficulty of performing differentiation has greatly increased and we are more dependent on computers to take derivates. The advantage of Automatic differentiation (AD) looms as it uses algorithms to efficiently and accurately evaluate derivatives of numerical functions. It can avoiding symbolic manipulation of functions while reaching an accuracy close to machine precision.

## Installation
Our package is for Python 3 only. To install autodiffCST, you need to have pip3 installed first. If you don't, please install pip3 following these instructions https://pip.pypa.io/en/stable/installing/.

Then, you could install this package by running 
```pip3 install autodiffCST``` from the command line. 

*Please be aware that you might need to manually install sympy and numpy prior to installing our package, depending on your environment. You could do so by running ```pip3 install sympy``` and ```pip3 install numpy``` in your terminal.*

An alternative is to clone our repository by running ```git clone https://github.com/auto-differentiaters-in-CST/cs107-FinalProject.git``` from the command line and then ```cd autodiffcst``` to go to the directory where the modules reside. Then use ```pip install -r requirements.txt``` to install the required pacakges.

## User Guide

After installation, you could import this package by ```import autodiffcst as cst```.

Then, you could initiate the VAD object by giving the point where you wish to differentiate. VAD can take in a vector input values, representing a point's coordinates in multi-dimensional space. Moreover, you could also try other supplementary features as in the code demo provided below. Notice that to use this demo, you can find a demo.ipynb file in the home directory.

``` python
# import modules
from autodiffcst import *

# base case: initialize VAD object with scalar values

[u] = VAD([5])           # initialize VAD objects u with a single point at 5

f = u*2-3                    # build a function with VAD object

print(f)                     # AD(value: [7], derivatives: [2.])

dfdu = f.diff(0)             # get derivative in the direction of u
print(dfdu)                  # 2.0

dfdu2 = f.diff([0,0],order=2) # get second derivative df^2/dudu
print(dfdu2)                  # 0.0

[x] = VAD([2],order=10)  # initialize as before, but specify that you want to get to order up tp 10

g = 2*exp(x)
print(g.higherdiff(10))             # 14.7781121978613

f = x**3                  # let's try another case for higher-order derivatives
print(f.higher)                  # array([12., 12.,  6.,  0.,  0.,  0.,  0.,  0.,  0.,  0.])
 
# Advanced cases: initialize VAD objects with vectors (multiple input values)

x, y, z = VAD([1,2,3])     # initialize VAD objects x, y, z with values 1, 2, 3 respectively
                                   # with multiple variable, you can skip brackets

f1,f2,f3= x+y, x**2+z, x*y*z   # build three functions with x, y, z
print(f3)                      # AD(value: [6], tag: [0 1 2], derivatives: [6. 3. 2.])  

print(jacobian(f1))

print(jacobian([f1, f2, f3]))

print(hessian(f3))

# Tricky case: using VAD to create a vector variable

v = VAD([1,2,3])        # initialize VAD objects: a vector v of value [1,2,3]
    
f = sin(v)              # build VAD: a single function applied to the vector v
print(f)                # print f's value and derivative. Here the second derivative will appear as a 3x3 matrix 

print(f.diff(0,order=1))          # get the first derivative with respect to v[0] (or x0), the first variable         

print(jacobian(f))

print(f.der2)                      # only in this case, you will be able to get the tensor hessian
```

## Broader Impact and Inclusivity Statement

### Broader Impact
We hope our package would be applied to different fields that require doing differentiations via computer programs: physics, engineering, applied mathematics, astronomy, and even other areas that the developers of this package have never imagined. We hope this package can be used to do automatic differentiations accurately and efficiently and can inspire the development of enhanced versions of automatic differentiation packages in the future. We see a number of possibilities that this package could be enhanced and would be happy to see them completed. 

On the other hand, we do not hope to see that this package is used for plagiarism, cheating, or shortcut for doing differentiation. The open-source nature of this package makes it accessible to people, but also susceptible to people who plan to use it for plagiarism. Users should be aware of this nature and wisely choose their way of using this package. This package is not designed for shortcuts of doing differentiation practices. People could use it to check their answers for calculating derivatives by hand or by other algorithms, but should not use it in place of derivative calculation practices. These practices have their purposes and using this package to get the answers does not contribute to the learning process.

We also see that when working on this project, we connected mathematical ideas such as Leibniz Rule and Faa di Bruno Formula to our automatic differentiation algorithms. Although this should not be the first time when people used these formulas to calculate higher-order derivatives, it was inspiring for us to do the implementation ourselves. We hope our project serves as a case where we bridge the gap between theories and applications. This experience will allow us and many students alike to keep striving for this goal and further tells that this is the best time when all kinds of knowledge come together to facilitate new discoveries.

### Software Inclusivity
The autodiffCST package and its developer welcome users who are contributors from all backgrounds and identities. We believe excellence in a collaborative project comes from trust, respect, and caring for each other, as it is evident through the process of developing this package. We tried our best to make our package as inclusive and user-friendly as possible with the willingness to reach more people that are interested in this package, by providing fitting documentation and instructions. Admittedly, this package is written in English and Python, but we welcome the contribution from people that are fluent in any language and programming languages. During the process of developing this package, pull requests are reviewed and approved by all developers. Whenever one of us feels the need to initiate a pull request, this person would communicate with other members and reach an agreement together. We would love to bring this positive communication to a future collaboration of this package and beyond.
