# This is an object-oriented program based on Python programming language, There is no special requirement for this program, JUST RUN:
import json
import os

# Define a class to represent a single transaction.
# In OOP, a class is like a blueprint for creating objects.
class Transaction:
    def __init__(self, title: str, amount: int, type: str, note=""):
        """
        Constructor for the Transaction class.
        :param title: Title or description of the transaction.
        :param amount: The monetary amount of the transaction.
        :param type: The type of transaction (e.g., "Expense" or "Deposit").
        :param note: An optional note for the transaction.
        """
        self.title = title
        self.amount = amount
        self.type = type
        self.note = note

    def display_info(self) -> str:
        """
        Returns a formatted string that displays the transaction information.
        """
        return (
            f"Transaction:\n"
            f"Title: {self.title}\n"
            f"Amount: {self.amount}\n"
            f"Type: {self.type}\n"
            f"Note: {self.note}"
        )


# Define a Bank class to represent a collection of transactions.
# This class encapsulates operations related to transactions.
class Bank:
    def __init__(self):
        """
        Constructor for the Bank class.
        Initializes an empty wallet to store Transaction objects.
        """
        self.wallet = []  # List to store transactions

    def add_transaction(self, transaction: Transaction):
        """
        Adds a new transaction to the wallet.
        :param transaction: A Transaction object to add.
        """
        self.wallet.append(transaction)

    def del_transaction(self, title: str) -> str:
        """
        Removes a transaction by its title.
        Iterates through the wallet and removes the transaction if found.
        :param title: Title of the transaction to remove.
        :return: A message indicating success or failure.
        """
        for trans in self.wallet:
            if trans.title == title:
                self.wallet.remove(trans)
                return f"'{title}' has been removed..."
        return f"The transaction '{title}' is not found..."

    def display(self) -> str:
        """
        Returns a string representing all transactions in the wallet.
        :return: A string of formatted transaction details.
        """
        if not self.wallet:
            return "No transaction available in your wallet!"
        # List comprehension: Loop over each transaction and get its display info.
        return "\n".join([trans.display_info() for trans in self.wallet])

    def search_wallet(self, query: str) -> str:
        """
        Searches the wallet for transactions that match the query in the title or type.
        :param query: Search string.
        :return: A string of matching transaction details or an error message.
        """
        # List comprehension: Filter transactions based on query (case-insensitive).
        found = [trans for trans in self.wallet if query.lower() in trans.title.lower() or query.lower() in trans.type.lower()]
        if not found:
            return "No transaction available!"
        return "\n".join([trans.display_info() for trans in found])

    def save_file(self, filename="wallet.json"):
        """
        Saves the transactions in the wallet to a JSON file.
        Uses list comprehension to create a list of dictionaries representing each transaction.
        :param filename: The name of the JSON file to save data to.
        """
        data = [
            {
                "Title": transaction.title,
                "Amount": transaction.amount,
                "Type": transaction.type,
                "Note": transaction.note,
            }
            for transaction in self.wallet  # List comprehension for simplicity
        ]
        with open(filename, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=4, ensure_ascii=False)  # ensure_ascii=False to preserve Persian text

    def load_file(self, file_name='wallet.json'):
        """
        Loads transactions from a JSON file and updates the wallet.
        :param file_name: The JSON file to load data from.
        """
        try:
            with open(file_name, "r", encoding="utf-8") as f:
                data = json.load(f)
                # List comprehension: Create Transaction objects from the JSON data.
                self.wallet = [
                    Transaction(trans["Title"], trans["Amount"], trans["Type"], trans["Note"])
                    for trans in data
                ]
        except FileNotFoundError:
            print("We don't have that file...")


def main():
    """
    Main function that runs the Personal Banking System.
    Provides a menu for the user to interact with the system.
    """
    wallet = Bank()
    while True:
        print("\n===== Personal Banking System =====")
        print("1. Add a Transaction")
        print("2. Remove a Transaction")
        print("3. Display all Transactions")
        print("4. Search for a Transaction")
        print("5. Save transactions to file")
        print("6. Load transactions from file")
        print("7. Exit")
        choice = input("Enter your choice (1-7): ")
        if choice == "1":
            title = input("Enter the title:\n>>> ")
            amount = int(input("Enter amount:\n>>> "))  # Convert input to integer
            type = input("Expense or Deposit?\n>>> ")
            note = input("Enter any note (optional):\n>>> ")
            transaction = Transaction(title, amount, type, note)
            wallet.add_transaction(transaction)
            print(f"'{title}' added successfully!")
        elif choice == "2":
            title = input("Enter the title:\n>>> ")
            print(wallet.del_transaction(title))
        elif choice == "3":
            print(wallet.display())
        elif choice == "4":
            query = input("Enter the title or type to search:\n>>> ")
            print(wallet.search_wallet(query))
        elif choice == "5":
            wallet.save_file()
            print("Saved as JSON!")
        elif choice == "6":
            wallet.load_file()
            print("Loaded JSON!")
        elif choice == "7":
            print("Exiting the program. Goodbye!")
            break
        else:
            print("Invalid choice.\nPlease try again.")


if __name__ == "__main__":
    main()
