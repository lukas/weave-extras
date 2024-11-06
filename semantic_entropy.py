import weave
import weaveindex
import scorers
import streamlit as st
import random
import pandas as pd

weave.init('example')


def semantic_entropy_buckets(question: str, answers: list[str]):
    # from paper: https://www.nature.com/articles/s41586-024-07421-0
    # github https://github.com/jlko/semantic_uncertainty
    buckets = []
    for answer in answers:
        bucketed = False
        for bucket in buckets:
            sample_answer = bucket[0]
            entailment1 = scorers.semantic_entailment(
                question, answer, sample_answer)
            entailment2 = scorers.semantic_entailment(
                question, sample_answer, answer)
            if entailment1 and entailment2:
                bucket.append(answer)
                bucketed = True
                break
        if not bucketed:
            buckets.append([answer])
    return buckets


def main():
    st.title("Semantic Entropy Analysis")

    # Add controls
    col1, col2 = st.columns(2)
    with col1:
        dataset_name = st.text_input("Dataset name", value="qa_pairs")
    with col2:
        n_answers = st.number_input("Number of answers", min_value=1, value=6)

    run_button = st.button("Run Analysis")

    if run_button:
        # Load QA pairs from Weave
        qa_pairs = weave.ref(dataset_name).get()
        qa_pairs_rows = qa_pairs.rows

        # Rest of the analysis
        random_row = qa_pairs_rows[random.randint(0, len(qa_pairs_rows)-1)]
        st.subheader("Random Question:")
        st.write(random_row['question'])

        vector_index = weaveindex.load_vector_index()
        query_engine = vector_index.as_query_engine()

        answers = weaveindex.query_multiple(
            random_row['question'],
            query_engine,
            model_name="gpt-4o-mini",
            n=n_answers
        )

        # Get buckets
        buckets = semantic_entropy_buckets(random_row['question'], answers)

        # Display buckets in tables
        st.subheader("Answer Buckets:")
        for i, bucket in enumerate(buckets, 1):
            st.write(f"Bucket {i}:")
            df = pd.DataFrame(bucket, columns=['Answers'])
            st.table(df)


if __name__ == "__main__":
    main()
