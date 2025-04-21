responses = {
    "hello": "Hello! How can I assist you today?",
    "hi": "Hi there! What can I help you with?",
    "how are you": "I'm doing well, thank you for asking!",
    "bye": "Goodbye! Have a great day.",
    "default": "I'm sorry, I didn't understand your request. Could you please rephrase it?"
}

print("Welcome to the chatbot! Type 'quit' to exit.")

while True:
    user_input = input("You: ").lower()

    # Check if the user wants to quit
    if user_input == "quit":
        break

    # Check if the user input matches any of the predefined responses
    if user_input in responses:
        print("Chatbot:", responses[user_input])
    else:
        print("Chatbot:", responses["default"])

print("Thank you for using the chatbot!")
