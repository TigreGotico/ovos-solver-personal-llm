# Personal LLM

this plugin uses langchain and llamaindex, it stores user facts and then uses a LLM to answer questions about them

it is similar in spirit to the "private GPT" projects floating around but instead of documents it works with sentences said by the user


# Usage

WIP

```python
from ovos_solver_personal_llm import UserInfo, PersonalLLMSolver

text_list = ["remember i have kept my keys in the bedroom drawer",
             "I need to go to shopping on saturday"]

for utt in text_list:
    # add facts to db  (it's a persistent .json file)
    UserInfo.remember(utt)

solver = PersonalLLMSolver()

# ask questions
response = solver.spoken_answer("Where did I keep my keys?")
print(response)
response = solver.spoken_answer("when is my next shopping date?")
print(response)
```