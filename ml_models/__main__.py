import os
from .baseline_manager import BaselineManager

PATH = os.path.join("assets", "LassoModel.pkl")

mgr = BaselineManager(PATH)

print("The model is ready to evaluate your question!")
print("Please enter the title of your question:")

title: str = input("Title: ")

print("Now enter the body of your question (you may also enter a"
      " path to a file that contains ir as html text):")

body: str = input("Question body: ")

if os.path.isfile(body):
    try:
        with open(body, "r") as f:
            contents = f.read()
        body = contents
    except:
        pass

prediction = mgr.make_prediction(title, body)
print("Predicted score is:", prediction)
