# Personal Banking System

This project is a simple personal banking system implemented in Python using Object-Oriented Programming (OOP) concepts. It allows users to add, remove, display, search, save, and load transactions.

## Overview

The system consists of two main classes:

1. **Transaction**:  
   - **Purpose**: Represents a single banking transaction.
   - **Attributes**:
     - `title`: A brief description of the transaction.
     - `amount`: The monetary amount involved.
     - `type`: Specifies whether the transaction is an "Expense" or "Deposit".
     - `note`: An optional note about the transaction.
   - **Methods**:
     - `display_info()`: Returns a formatted string showing the transaction details.

2. **Bank**:
   - **Purpose**: Manages a collection of transactions (a wallet) and provides methods to manipulate them.
   - **Methods**:
     - `add_transaction(transaction)`: Adds a new transaction to the wallet.
     - `del_transaction(title)`: Deletes a transaction by its title.
     - `display()`: Returns a string with details of all transactions.
     - `search_wallet(query)`: Searches for transactions that match a given query in the title or type.
     - `save_file(filename)`: Saves the wallet (list of transactions) to a JSON file.
     - `load_file(file_name)`: Loads transactions from a JSON file into the wallet.

## Key OOP Concepts Demonstrated

- **Encapsulation**:  
  The `Transaction` and `Bank` classes encapsulate their data (attributes) and behaviors (methods), which helps to organize code and manage complexity.

- **Abstraction**:  
  The classes hide the internal details of transaction management and provide simple interfaces (methods) for interacting with the transactions.

- **Reusability**:  
  The methods in the classes (e.g., `display_info`, `add_transaction`, `del_transaction`) are designed to be reused, reducing code duplication.

- **List Comprehensions**:  
  Used in the `save_file` and `load_file` methods to create or process lists in a concise and readable manner. For example, a list comprehension is used to convert each transaction object into a dictionary:
  ```python
  data = [
      {
          "Title": transaction.title,
          "Amount": transaction.amount,
          "Type": transaction.type,
          "Note": transaction.note,
      }
      for transaction in self.wallet
  ]
