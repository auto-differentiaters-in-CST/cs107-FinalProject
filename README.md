# cs107-FinalProject
## This is the repository for CS107 group project for fall 2020 at HU.
### Group 5
Contributors: Xiaohan Yang, Max Li, Runting Yang,Hanwen Zhang

travis badge:
[![Build Status](https://travis-ci.com/auto-differentiaters-in-CST/cs107-FinalProject.svg?token=AjVcVSqkqdiJgwaimWYR&branch=master)](https://travis-ci.com/auto-differentiaters-in-CST/cs107-FinalProject)

codecov badge:
[![codecov](https://codecov.io/gh/auto-differentiaters-in-CST/cs107-FinalProject/branch/master/graph/badge.svg?token=US1Y8Z9OE0)](https://codecov.io/gh/auto-differentiaters-in-CST/cs107-FinalProject)


## Introduction
This package autodiffCST implements automatic differentiation. It would be used to automatically differentiate functions via forward mode and reverse mode, depending on the user's choice. It would also provides an option of performing second order differentiation. 

Differentiation, namely, the process of finding the derivatives of functions, is very prevalent in various areas of science and engineering. It can often be used to find the extrema of functions with single or multiple variables. With the advance of technology, more complicated functions and larger dataset are developed. The difficulty of performing differentiation has greatly increased and we are more dependent on computers to take derivates. Nowadays, we have three major ways of performing differentiation: symbolic, numerical and automatic (algorithmic) differentiation. We will focus on automatic differentiation for the rest of this document.

**Note: The current version could only implement the first order differentiation for one function with multiple variables.**

## Installation

Our package is for Python 3 only. The current version is not released on PyPI yet, so please clone this repository to use our package. 

* Clone the package repository: ```git clone https://github.com/auto-differentiaters-in-CST/cs107-FinalProject.git```
* Change into the new directory: ```cd cs107-FinalProject``` .
* Install the dependencies using ```pip install -r requirements.txt```.

## User Guide

After installation, users could import this package by ```from autodiffcst import AD```.

Then, they could simply initiate the AD object by giving the point where they wish to differentiate. Moreover, they could also try other supplementary features as in the code demo provided below.

``` python
# import modules
from autodiffcst import AD as ad
from autodiffcst import admath as admath

# base case: initialize AD object with scalar values

x = ad.AD(5, tag = "x") # initialize AD object called "x" with the value 5
y = ad.AD(3, tag = "y") # initialize AD object called "y" with the value 3

f1 = x*y              # build a function with AD objects, the function will also be an AD object
print(f1)             # print AD(value: {15}, derivatives: {'x': 3, 'y': 5})

dfdx = f1.diff("x") # returns the derivative with respect to x
print(dfdx)                  # print 3
 
jacobian = ad.jacobian(f1) # returns a gradient vector of f
print(jacobian)  # print [5,3]

f2 =  admath.sin(x) + y   # build a function with AD objects, the function will also be an AD object
print(f2)             # print AD(value: {15}, derivatives: {'x': 3, 'y': 5})

dfdx = f.diff("x") # returns the derivative with respect to x
print(dfdx)                  # print 3
 
jacobian = ad.jacobian(f) # returns a gradient vector of f
print(jacobian)  # print [5,3]
```