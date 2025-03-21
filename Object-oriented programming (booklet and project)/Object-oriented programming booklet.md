# Object-Oriented Programming (OOP) in Python

This booklet introduces the fundamentals of object-oriented programming (OOP) using Python. OOP organizes code by grouping related variables (properties) and functions (methods) into objects. Throughout this guide, you will learn the basics of creating classes and objects, understanding the differences between instance and class attributes, and using special methods (such as constructors, getters, setters, and more). We also cover key OOP concepts including encapsulation, inheritance, polymorphism, and abstraction.

---

## Table of Contents

1. [Introduction to OOP](#introduction-to-oop)
2. [Basic OOP Concepts](#basic-oop-concepts)
3. [Creating Classes and Instantiating Objects](#creating-classes-and-instantiating-objects)
4. [Constructors: The `__init__` Method](#constructors-the-__init__-method)
5. [Instance vs. Class Attributes](#instance-vs-class-attributes)
6. [Methods: Instance, Static, and Class Methods](#methods-instance-static-and-class-methods)
7. [Inheritance](#inheritance)
8. [Encapsulation and Getters/Setters](#encapsulation-and-getterssetters)
9. [Polymorphism and Abstraction](#polymorphism-and-abstraction)
10. [Advanced Examples and CSV Usage](#advanced-examples-and-csv-usage)

---

## Introduction to OOP

Object-oriented programming (OOP) is a programming paradigm that organizes code into objects. An **object** is an instance of a class, and it groups together data (called properties) and functions (called methods) that operate on that data. OOP helps make code modular, reusable, and easier to understand.

### Key Terms

- **Class**: A blueprint for creating objects. It defines a set of attributes and methods.
- **Object (Instance)**: A concrete instance of a class.
- **Properties (Attributes)**: Variables that hold data about an object.
- **Methods**: Functions defined inside a class that describe the behaviors of an object.
- **Encapsulation**: The practice of restricting direct access to some of an object's components.
- **Inheritance**: A mechanism where one class (child) inherits attributes and methods from another class (parent).
- **Polymorphism**: The ability to use a common interface for different underlying data types.
- **Abstraction**: Hiding complex implementation details and showing only the necessary parts of an object.

---

## Basic OOP Concepts

### Objects and Classes

Consider a car object. A car has:
- **Properties**: `make`, `model`, `color`
- **Methods**: `start()`, `stop()`, `move()`

Even basic data types in Python (such as `int`, `float`, and `str`) are objects instantiated from built-in classes.

#### Examples of Instantiated Objects

```python
1.94             # A float object
123              # An int object
'Hello, World!'  # A str object

class Item:  # Class names should follow CamelCase
    def calculate_total_price(self, x, y):
        return x * y

# Creating instances (objects) of the class
item1 = Item()

# Dynamically assigning properties (not recommended for consistency)
item1.name = "Phone"
item1.price = 100
item1.quantity = 5

# Using the method to calculate the total price
print(item1.calculate_total_price(item1.price, item1.quantity))

class Item:
    def __init__(self, name: str, price: int, quantity: int, guarantee_month=0):
        """Constructor that initializes object properties"""
        print(f"An instance created with name: {name}")
        
        # Input validation using assert statements
        assert price >= 0, f"Price '{price}' must be greater than or equal to zero"
        assert quantity >= 0, f"Quantity '{quantity}' must be greater than or equal to zero"
        
        # Assign properties to the instance (self)
        self.name = name
        self.price = price
        self.quantity = quantity
    
    def calculate_total_price(self):
        """Method to calculate the total price of an item"""
        return self.price * self.quantity

# Creating instances with the constructor
item1 = Item(name="Phone", price=100, quantity=5)
item2 = Item(name="Laptop", price=1000, quantity=3)

# Accessing object properties and calling methods
print(f"{item1.name}\n{item2.name}")
print(item1.calculate_total_price())
print(item2.calculate_total_price())

class Item:
    # Class attributes
    pay_rate = 0.8  # 20% discount applied by default
    all = []        # A list to store all instances of the class
    
    def __init__(self, name: str, price: int, quantity: int, guarantee_month=0):
        print(f"An instance created with name: {name}")
        
        # Validation
        assert price >= 0, f"Price '{price}' must be greater than or equal to zero"
        assert quantity >= 0, f"Quantity '{quantity}' must be greater than or equal to zero"
        
        # Instance attributes
        self.name = name
        self.price = price
        self.quantity = quantity
        
        # Append the instance to the class-level list
        Item.all.append(self)
    
    def calculate_total_price(self):
        """Calculates the total price of the item."""
        return self.price * self.quantity

    def apply_discount(self):
        """Applies a discount to the item's price."""
        self.price = self.price * self.pay_rate

    def __repr__(self):
        """Returns a string representation of the object."""
        return f"Item({self.name}, {self.price}, {self.quantity})"

# Example demonstrating attribute access
item1 = Item('Phone', 100, 1)
item1.pay_rate = 0.2  # Overwriting the discount at the instance level
item1.apply_discount()
print(item1.price)

# Print the class and instance dictionaries for comparison
print(Item.__dict__)
print(".......................................................................")
print(item1.__dict__)

# Creating more items
item2 = Item('Laptop', 100, 3)
item2.pay_rate = 0.9  # Changing discount rate for this instance
item2.apply_discount()
print(item2.price)

item3 = Item('Cable', 10, 50)
item4 = Item('Mouse', 50, 5)
item5 = Item('Keyboard', 75, 5)

print(Item.all)

class MathUtils:
    @staticmethod
    def add(a, b):
        return a + b

print(MathUtils.add(3, 5))  # Output: 8
# Inline: add() is static because it doesn't use self or cls. It's logically related to MathUtils.

class User:
    def __init__(self, name, age):
        self.name = name
        self.age = age

    @classmethod
    def from_string(cls, user_str):
        name, age = user_str.split(",")
        return cls(name, int(age))  # Creates and returns a new instance

user1 = User.from_string("Alice,25")
print(user1.name, user1.age)  # Output: Alice 25
# Inline: from_string is an alternative constructor that parses a string to create a User.

class Item:
    pay_rate = 0.8
    all = []
    
    def __init__(self, name: str, price: int, quantity: int, guarantee_month=0):
        print(f"An instance created with name: {name}")
        assert price >= 0, f"Price '{price}' must be greater than or equal to zero"
        assert quantity >= 0, f"Quantity '{quantity}' must be greater than or equal to zero"
        self.name = name
        self.price = price
        self.quantity = quantity
        Item.all.append(self)
    
    def calculate_total_price(self):
        return self.price * self.quantity
    
    def apply_discount(self):
        self.price = self.price * self.pay_rate
    
    def __repr__(self):
        return f"Item({self.name}, {self.price}, {self.quantity})"

class Phone(Item):  # Inheriting from Item
    def __init__(self, name: str, price: int, quantity: int, broken_phone=0):
        # Call the parent class constructor
        super().__init__(name, price, quantity)
        assert broken_phone >= 0, f"Broken phone {broken_phone} is not valid"
        self.broken_phone = broken_phone
        # Note: If Phone has its own collection of instances, define Phone.all separately.
        # For this example, we assume Phone instances are tracked in Item.all.

# Create an instance of Phone
phone1 = Phone("jscPhone10", 500, 5, 1)
print(Item.all)
# Inline: The Phone class inherits from Item. The use of super() calls the parent’s __init__.

import csv

class Item:
    pay_rate = 0.8
    all = []
    
    def __init__(self, name: str, price: int, quantity=0, guarantee_month=0):
        assert price >= 0, "Price must be greater than zero"
        assert quantity >= 0, "Quantity must be greater than zero"
        self._name = name               # Conventionally "protected"
        self.__set_price(price)         # Use a private method to set price
        self.quantity = quantity
        Item.all.append(self)
    
    @property
    def price(self):
        return self.__price  # Getter returns the mangled private attribute __price
    
    def __set_price(self, price):
        """Private method to set price using name mangling.
        The double underscore in __price triggers name mangling.
        """
        self.__price = price
    
    def apply_discount(self):
        # Using self.pay_rate allows instance-level overriding
        self.__price = self.__price * self.pay_rate
    
    def apply_increment(self, increment_value):
        self.__price += self.__price * increment_value 

    def __connect(self, smtp_server):
        # Private method for connecting to an SMTP server (implementation hidden)
        pass

    def __prepare_body(self):
        # Private method to prepare the email body
        return f"Hello,\nWe have {self._name} with quantity {self.quantity}\nRegards, Shadow"
    
    def send_email(self):
        # Abstracting the email sending process
        self.__connect("smtp.server.com")
        body = self.__prepare_body()
        self.__send()  # This would be a private method handling the actual sending
    
    @property
    def name(self):
        """Getter for name. Uses the mangled __name attribute if set."""
        return self.__name  # Using __name to store the actual name
    
    @name.setter
    def name(self, value):
        """Setter for name with validation.
        Inline: The setter validates that the name length is acceptable.
        """
        if len(value) > 10:
            raise Exception("The name is too long")
        else:
            self.__name = value  # Assign the value to the mangled __name

    def calculate_total_price(self):
        return self.price * self.quantity 

    @classmethod
    def instantiate_from_csv(cls):
        # Class method to instantiate objects from a CSV file
        with open("new.csv", "r") as f:
            reader = csv.DictReader(f)
            items = list(reader)
        for item in items:
            print(
                "name =", item.get("name"),
                "price =", float(item.get("price")),
                "quantity =", float(item.get("quantity"))
            )
    
    def __repr__(self):
        return f"Item({self._name}, {self.__price}, {self.quantity})"

# Inline Explanations:
# - The comment "why do we use __price what are those __?" is answered above: the double underscore triggers name mangling for private attributes.
# - The differences between _name and __name are that __name is harder to access from outside the class, reinforcing encapsulation.
# - The @property decorator creates a getter, and the @name.setter creates a setter for the name attribute.
# - "static", "class", and "init" are explained in earlier sections.
# - The "getter" is a method to read a property, and a "setter" is a method to set a property, often with validation.

# Demonstrate getter and setter usage
try:
    item1 = Item("ShortName", 750, 5)
    item1.apply_increment(0.2)
    print(item1.price)
except Exception as e:
    print(e)

# Assume that Item is imported from a separate module
# from item import Item

class Keyboard(Item):
    def __init__(self, name: str, price: int, quantity=0):
        super().__init__(name, price, quantity)
        # Additional initialization specific to Keyboard can go here

    def apply_increment(self, increment_value):
        # Override to show that we can modify behavior for a subclass
        super().apply_increment(increment_value)

# Using Keyboard with encapsulated properties
keyboard_item = Keyboard("MyItem", 750)
keyboard_item.apply_increment(0.2)  # Increase price by 20%
print(keyboard_item.price)

# Import statements would normally be at the top of your module
# from item import Item
# from new import Keyboard   # If Keyboard were defined in another module

# Inheriting from Item and adding specific functionality for Phone
class Phone(Item):
    def __init__(self, name: str, price: int, quantity: int, broken_phone=0):
        super().__init__(name, price, quantity)
        assert broken_phone >= 0, f"Broken phone {broken_phone} is not valid"
        self.broken_phone = broken_phone
        # Optionally, you can keep a separate collection for Phone instances if needed:
        # Phone.all.append(self)

# Create an instance of Phone
phone1 = Phone("jscPhone10", 500, 5, 1)
print(Item.all)
# Inline: This demonstrates inheritance where Phone extends Item.

# Demonstrate read-only attributes with getters and setters using Keyboard
class Keyboard(Item):
    def __init__(self, name: str, price: int, quantity=0):
        super().__init__(name, price, quantity)

    def apply_increment(self, increment_value):
        # This method uses the encapsulated __price to update the price
        self.apply_increment(increment_value)

# Create a Keyboard instance and apply an increment to its price
keyboard_item = Keyboard("Keyboard", 750)
keyboard_item.apply_increment(0.2)
print(keyboard_item.price)
