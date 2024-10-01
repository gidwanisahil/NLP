import nltk
from nltk.chat.util import Chat, reflections

# Define pairs of input and responses
pairs = [
    [
        r"my name is (.*)",
        ["Hello %1, how can I help you today?",]
    ],
    [
        r"hi|hello|hey",
        ["Hello! How can I assist you?",]
    ],
    [
        r"what is your name?",
        ["I am a chatbot created to help you.",]
    ],
    [
        r"how are you?",
        ["I'm just a program, but thanks for asking!",]
    ],
    [
        r"sorry (.*)",
        ["No problem! How can I assist you?",]
    ],
    [
        r"quit",
        ["Bye! Take care.",]
    ],
    [
        r"(.*)",
        ["I'm sorry, I don't understand that. Can you rephrase?",]
    ]
]

def chatbot():
    print("Hi! I'm a chatbot. Type 'quit' to exit.")
    chat = Chat(pairs, reflections)
    chat.converse()

if __name__ == "__main__":
    chatbot()
