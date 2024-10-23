import re
file = open("inference.txt", "r")
content = file.read()

def preprocess_case(text):
    text = re.sub(r'"', '', text)
    # tokens = text.split(' ')
    print(text)

preprocess_case(content)

