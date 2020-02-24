# blogentry: https://medium.com/predict/creating-a-chatbot-from-scratch-using-keras-and-tensorflow-59e8fc76be79
# notebook: https://colab.research.google.com/drive/1FKhOYhOz8d6BKLVVwL1YMlmoFQ2ML1DS#scrollTo=imkdw4os6FI4&forceEdit=true&sandboxMode=true
# datasource: https://github.com/shubham0204/Dataset_Archives

import os
import re
import yaml

def load_data_from_yml(file_dir, parse_attr):
    '''
    Assume different yml files with different topics.
    QA pairs come in lines after "conversations:"
    "--": Question (input) / "  -" Answer (output)
    '''
    questions = list()
    answers = list()

    file_names = os.listdir(file_dir)
    for file_name in file_names:
        try:
            with open(f"{file_dir}/{file_name}") as f:
                input_data = yaml.safe_load(f)
            for obj in input_data[parse_attr]:
                if len(obj) > 2:  # tmp. skip if more than one answer
                    continue

                # obj[0] = obj[0].lower()
                # obj[1] = obj[1].lower()
                # obj[0] = " ".join(re.sub(r'[\.,!?-]+', ' ', obj[0]).split())
                # obj[1] = " ".join(re.sub(r'[\.,!?-]+', ' ', obj[1]).split())

                if obj[0] in questions:  # tmp. skip if question already existing
                    continue

                questions.append(obj[0])
                answers.append(obj[1])
        except Exception:
            continue
    return questions, answers

def load_data_with_intent_from_yml(file_dir, parse_attr):
    '''
    Assume different yml files with different topics.
    QA pairs come in lines after "conversations:"
    "--": Question (input) / "  -" Answer (output)
    '''
    questions = list()
    answers = list()

    file_names = os.listdir(file_dir)
    for file_name in file_names:
        try:
            with open(f"{file_dir}/{file_name}") as f:
                input_data = yaml.safe_load(f)
            intent = "_default_"
            for obj in input_data["categories"]:
                intent = f"_{obj[0]}_"
            for obj in input_data[parse_attr]:
                if len(obj) > 2:  # tmp. skip if more than one answer
                    continue

                # obj[0] = obj[0].lower()
                # obj[1] = obj[1].lower()
                # obj[0] = " ".join(re.sub(r'[\.,!?-]+', ' ', obj[0]).split())
                # obj[1] = " ".join(re.sub(r'[\.,!?-]+', ' ', obj[1]).split())

                if obj[0] in questions:  # tmp. skip if question already existing
                    continue

                questions.append(f"{intent} {obj[0]}")
                answers.append(f"{intent} {obj[1]}")
        except Exception:
            continue
    return questions, answers

def tag_answers(answers):
    for i in range(len(answers)):
        answers[i] = f"<BOS> {answers[i]} <EOS>"
    return answers

# questions, answers = load_data_from_yml("../../data/chatbot_data/shubham0204/chatbot_nlp", "conversations")
# answers = tag_answers(answers)
