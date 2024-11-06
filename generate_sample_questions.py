import openai
from pydantic import BaseModel
import json
import sys
from weave import Dataset
from pathlib import Path
from typing import List, Dict
import weave
import streamlit as st


class QAPair(BaseModel):
    question: str
    answer: str


class QAPairList(BaseModel):
    qa_pairs: list[QAPair]


def generate_qa_pairs(document_text: str, num_pairs: int = 3) -> QAPairList:
    """
    Generate question-answer pairs from a document using OpenAI's API.

    Args:
        document_text: The text content of the document
        num_pairs: Number of Q&A pairs to generate (default: 3)

    Returns:
        List of dictionaries containing question-answer pairs
    """
    prompt = f"""
    Please generate {num_pairs} question-answer pairs based on the following text. 
    Return only a JSON array where each object has 'question' and 'answer' fields.
    
    Text: {document_text}
    """

    try:
        response = openai.beta.chat.completions.parse(
            model="gpt-4o",
            messages=[
                {"role": "system", "content": "You are a helpful assistant that generates question-answer pairs in JSON format."},
                {"role": "user", "content": prompt}
            ],
            response_format=QAPairList
        )
        # Parse the JSON response
        qa_pairs = response.choices[0].message.parsed
        return qa_pairs

    except Exception as e:
        breakpoint()
        print(f"Error generating Q&A pairs: {e}")
        return []


def process_documents(doc_directory: str) -> QAPairList:
    """
    Process all documents in a directory and generate Q&A pairs for each.

    Args:
        doc_directory: Path to directory containing documents

    Returns:
        Dictionary mapping document names to their Q&A pairs
    """
    results = {}
    doc_path = Path(doc_directory)

    for file_path in doc_path.glob('*.txt'):  # Adjust file pattern as needed
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                document_text = f.read()

            qa_pairs = generate_qa_pairs(document_text)
            results[file_path.name] = qa_pairs

        except Exception as e:
            print(f"Error processing {file_path}: {e}")

    return results


def process_document(file_path: str) -> QAPairList:
    with open(file_path, 'r', encoding='utf-8') as f:
        document_text = f.read()

    return generate_qa_pairs(document_text)


def generate_qa_pair_dataset(inp_dataset_name: str, out_dataset_name: str, num_pairs: int = 3, content_column_name: str = 'content') -> QAPairList:
    # Retrieve the dataset
    qa_results = []
    dataset = weave.ref(inp_dataset_name).get()
    for row in dataset.rows:
        qa_pairs = generate_qa_pairs(
            row[content_column_name], num_pairs=num_pairs)
        for qa_pair in qa_pairs.qa_pairs:
            qa_results.append({
                'filename': row['filename'],
                'question': qa_pair.question,
                'answer': qa_pair.answer
            })

    qa_result_dataset = Dataset(name=out_dataset_name, rows=qa_results)
    weave.publish(qa_result_dataset)


def main():
    st.title("Question-Answer Pair Generator")

    # Add input fields
    inp_dataset_name = st.text_input("Input Dataset Name", "documents")
    out_dataset_name = st.text_input("Output Dataset Name", "qa_pairs")
    num_pairs = st.number_input("Number of Q&A Pairs", min_value=1, value=3)

    # Add a button to trigger generation
    if st.button("Generate Q&A Pairs"):
        with st.spinner("Generating Q&A pairs..."):
            generate_qa_pair_dataset(
                inp_dataset_name, out_dataset_name, num_pairs)
        st.success(f"Successfully generated {num_pairs} Q&A pairs!")


weave.init('example')

if __name__ == "__main__":
    main()

# if __name__ == "__main__":

#     # Directory containing your documents
#     DOC_DIRECTORY = sys.argv[1]

#     # Generate Q&A pairs for all documents
#     all_qa_pairs = process_document(DOC_DIRECTORY)

#     # Save results to a JSON file
#     with open("qa_pairs_output.json", "w", encoding="utf-8") as f:
#         json.dump(all_qa_pairs["qa_pairs"].model_dump(),
#                   f, indent=2, ensure_ascii=False)
