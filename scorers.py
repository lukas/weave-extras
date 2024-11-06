
import openai
import weave
from pydantic import BaseModel


class LLMScore(BaseModel):
    score: int
    explanation: str


class SemanticEntailment(BaseModel):
    entailment: bool
    explanation: str


async def llm_judge_question_answer_match(question: str, answer: str, model_output: dict) -> LLMScore:
    # from https://www.nature.com/articles/s41586-024-07421-0

    prompt = f"""We are assessing the quality of answers to the following question: {question}

    The expected answer is : {answer}

    The proposed answer is: {model_output['pred']}

    Within the context of the question, does the proposed answer mean the same as the expected answer? 
    Respond with a 0 if the answer is not the same as the expected answer, and a 1 if the answer is 
    the same as the expected answer.
    """
    response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
              "content": "You are an expert at evaluating answer similarity."},
            {"role": "user", "content": prompt}
        ],
        response_format=LLMScore
    )
    score = response.choices[0].message.parsed.score
    return score


@weave.op()
def semantic_entailment(question: str, answer1: str, answer2: str) -> float:
    # returns a score between 0 and 1 depending on how much information the answer contains

    prompt = f"""We are evaluating answers to the question {question}

    Here are two possible answers:

    Possible Answer 1: {answer1}

    Possible Answer 2: {answer2}

    Does Possible Answer 1 semantically entail Possible Answer 2? Respond with true or false.
    """
    response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
                "content": "You are an expert at evaluating answer similarity."},
            {"role": "user", "content": prompt}
        ],
        response_format=SemanticEntailment
    )
    entailment = response.choices[0].message.parsed.entailment
    return entailment


@weave.op()
def llm_judge_match_score(answer: str, model_output: dict) -> LLMScore:
    # returns a score between 1 and 5 depending on how similar the answer and model output are

    prompt = f"""Compare the following answer and model output for semantic similarity and correctness.
    
    Reference Answer: {answer}
    Model Output: {model_output['pred']}
    
    Rate the match on a scale of 1-5 where:
    1 = Completely different/incorrect
    2 = Major differences/errors
    3 = Partially correct with some differences
    4 = Mostly correct with minor differences
    5 = Excellent match/essentially identical
    
    Return your response as a JSON object with two fields:
    - score: integer between 1-5
    - explanation: string explaining the rating
    
    Response format:
    {{"score": <number>, "explanation": "<explanation>"}}
    """

    response = openai.beta.chat.completions.parse(
        model="gpt-4o-mini",
        messages=[
            {"role": "system",
                "content": "You are an expert at evaluating answer similarity."},
            {"role": "user", "content": prompt}
        ],
        response_format=LLMScore
    )
    score = response.choices[0].message.parsed.score
    return score
