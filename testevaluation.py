#MCQ Evaluation

def evaluate_mcq(student_answers, correct_answers):
    score = 0
    for s, c in zip(student_answers, correct_answers):
        if s == c:
            score += 1
    return score

# Example
student = ['A', 'C', 'B', 'D', 'A']
answer_key = ['A', 'B', 'B', 'D', 'A']
print("MCQ Score:", evaluate_mcq(student, answer_key), "/", len(answer_key))


#Coding Assignment Grading (Using Test Cases)

def evaluate_code_submission(student_code, test_cases):
    exec_globals = {}
    try:
        exec(student_code, exec_globals)
    except Exception as e:
        return 0, f"Compilation Error: {e}"
    
    passed = 0
    for case in test_cases:
        try:
            output = exec_globals[case['function']](*case['input'])
            if output == case['expected']:
                passed += 1
        except Exception as e:
            continue
    return passed, f"{passed}/{len(test_cases)} test cases passed"

# Example function to sum two numbers
student_code = """
def add(a, b):
    return a + b
"""

test_cases = [
    {'function': 'add', 'input': (2, 3), 'expected': 5},
    {'function': 'add', 'input': (-1, 1), 'expected': 0}
]

score, msg = evaluate_code_submission(student_code, test_cases)
print("Coding Score:", msg)


