import streamlit as st
from bias import bias_list, openai_moderation_list, toxicity_list


def main():
    st.title("Bias Detection Tool")

    # Add multiselect for analysis types
    analysis_types = st.multiselect(
        "Select analysis types:",
        ["Bias Detection",
         "Content Moderation (OpenAI)",
         "Toxicity Detection"],
        default=["Bias Detection"]
    )

    # Create a text input for the query
    query = st.text_area("Enter your text to analyze:",
                         height=150,
                         placeholder="Type or paste your text here...")

    # Add a button to trigger the analysis
    if st.button("Analyze"):
        if query:
            try:
                # Run selected analyses
                if "Bias Detection" in analysis_types:
                    st.write("Bias Detection")
                    bias_result = bias_list(query)

                    # Display bias results
                    if bias_result.issues:
                        st.subheader("Detected Biases")
                        bias_data = [
                            {
                                "Issue": issue.issue,
                                "Score": issue.score,
                                "Explanation": issue.explanation
                            }
                            for issue in bias_result.issues
                        ]
                        st.dataframe(
                            bias_data,
                            column_config={
                                "Issue": st.column_config.TextColumn(
                                    "Detected Issue",
                                    width="medium",
                                ),
                                "Score": st.column_config.NumberColumn(
                                    "Score",
                                    width="small",
                                ),
                                "Explanation": st.column_config.TextColumn(
                                    "Explanation",
                                    width="large",
                                )
                            },
                            hide_index=True,
                        )
                    else:
                        st.success("No biases detected in the text.")

                if "Content Moderation (OpenAI)" in analysis_types:
                    st.write("Content Moderation (OpenAI)")

                    moderation_result = openai_moderation_list(query)

                    # Display moderation results
                    if moderation_result.issues:
                        st.subheader("Content Moderation Results")
                        moderation_data = [
                            {
                                "Issue": issue.issue,
                                "Score": issue.score,
                                "Explanation": issue.explanation
                            }
                            for issue in moderation_result.issues
                        ]
                        st.dataframe(
                            moderation_data,
                            column_config={
                                "Issue": st.column_config.TextColumn(
                                    "Category",
                                    width="medium",
                                ),
                                "Score": st.column_config.NumberColumn(
                                    "Score",
                                    width="small",
                                ),
                                "Explanation": st.column_config.TextColumn(
                                    "Explanation",
                                    width="large",
                                )
                            },
                            hide_index=True,
                        )
                    else:
                        st.success(
                            "No content moderation detected in the text.")

                if "Toxicity Detection" in analysis_types:
                    st.write("Toxicity Detection")
                    toxicity_result = toxicity_list(query)

                    # Display toxicity results
                    if toxicity_result.issues:
                        st.subheader("Toxicity Detection Results")
                        toxicity_data = [
                            {
                                "Issue": issue.issue,
                                "Score": issue.score,
                                "Explanation": issue.explanation
                            }
                            for issue in toxicity_result.issues
                        ]
                        st.dataframe(
                            toxicity_data,
                            column_config={
                                "Issue": st.column_config.TextColumn(
                                    "Category",
                                    width="medium",
                                ),
                                "Score": st.column_config.NumberColumn(
                                    "Score",
                                    width="small",
                                ),
                                "Explanation": st.column_config.TextColumn(
                                    "Explanation",
                                    width="large",
                                )
                            },
                            hide_index=True,
                        )
                    else:
                        st.success("No toxicity detected in the text.")

            except Exception as e:
                st.error(f"An error occurred: {str(e)}")
        else:
            st.warning("Please enter some text to analyze.")


if __name__ == "__main__":
    main()
