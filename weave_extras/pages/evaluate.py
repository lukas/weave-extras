import streamlit as st
import weave
import weaveindex
import model
import asyncio
from pydantic import BaseModel
from scorers import llm_judge_question_answer_match
from bias import bias_score
from weave import Evaluation
import pandas as pd


class LLMScore(BaseModel):
    score: int
    explanation: str


st.title("Model Evaluation")

# Initialize Weave and load models
weave.init('example')
qa_pairs: weave.Dataset = weave.ref("qa_pairs").get()

# Load the vector index
vector_index = weaveindex.load_vector_index()
query_engine = vector_index.as_query_engine()

# Add checkbox for testing mode before model selection
test_mode = st.checkbox("Test on small dataset (3 rows)", value=False)

# Replace single model selection with multiselect
selected_models = st.multiselect(
    "Select OpenAI Models",
    ["gpt-3.5-turbo", "gpt-4", "gpt-4-turbo", "gpt-4o-mini", "gpt-4o"],
    default=["gpt-3.5-turbo"]
)

# Add this after the model selection multiselect and before the "Run Evaluation" button
available_evaluators = {
    "Question-Answer Match": llm_judge_question_answer_match,
    "Bias Score": bias_score
}

selected_evaluators = st.multiselect(
    "Select Evaluators",
    options=list(available_evaluators.keys()),
    default=["Question-Answer Match"],
    help="Choose which evaluation metrics to run"
)

if st.button("Run Evaluation"):
    if not selected_models:
        st.warning("Please select at least one model to evaluate.")
    elif not selected_evaluators:
        st.warning("Please select at least one evaluator.")
    else:
        with st.spinner("Evaluating models..."):
            all_results = {}

            # Add dataset selection based on test mode
            # if test_mode else qa_pairs
            if test_mode:
                evaluation_dataset = qa_pairs.rows[:3]
            else:
                evaluation_dataset = qa_pairs

            # Create all model instances first
            model_instances = {
                model_name: model.QAModel(
                    query_engine=query_engine, model_name=model_name)
                for model_name in selected_models
            }

            # Create a placeholder for the table
            table_placeholder = st.empty()

            # Single evaluation for all models
            evaluation = Evaluation(
                dataset=evaluation_dataset,
                scorers=[available_evaluators[eval_name]
                         for eval_name in selected_evaluators],
            )

            # Evaluate models one at a time
            with st.status("Running evaluation...") as status:
                for model_name, model_instance in model_instances.items():
                    status.update(label=f"Evaluating {model_name}...")
                    result = asyncio.run(evaluation.evaluate(model_instance))
                    all_results[model_name] = result

                    # Update table after each model evaluation
                    table_data = []
                    all_metrics = set()
                    for completed_results in all_results.values():
                        all_metrics.update(completed_results.keys())

                    for completed_model, results in all_results.items():
                        row = {'Model': completed_model}
                        for metric in all_metrics:
                            if metric in results:
                                row[metric] = f"{results[metric]['mean']:.2f}"
                            else:
                                row[metric] = "N/A"
                        table_data.append(row)

                    # Update the table in place
                    table_placeholder.table(table_data)

            st.success("Evaluation completed!")

# Define any custom scoring function
