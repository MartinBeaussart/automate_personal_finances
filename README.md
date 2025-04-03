# Automate personal finances

**Overview**
-----------

This is a personal project aimed at categorizing my monthly personal finance transactions into categories such as Groceries, Rent, Transport, and more.

**Current Status**
-----------------

The project is currently in its early stages, with a relatively simple model using Bert that categorizes transactions. See future plans

**Getting Started**
-------------------

To get started with the project, clone this repository and then add your data.

- In data/input/ the csv you want to categorize with inside at least a column 'Description'
- In data/raw_data/train/ multiples csv you want to be part of the training with inside at least a column 'Description' and 'Category'
- In data/raw_data/test/ multiples csv you want to be part of the testing with inside at least a column 'Description' and 'Category'

Finaly run the `main.py` script in the root directory. 

**Future Plans**
----------------

* Developing more complex models that use not only the descriptionas input but also the time or the amount
* Try sentence model that would fit better than Bert
* Generating synthetic data to augment the existing dataset
* Automating the preprocessing of transaction data downloaded from banks

