import weaveindex
import weave
import streamlit as st

docs_dir = weaveindex.temp_dir


vector_index = weaveindex.load_vector_index()
query_engine = vector_index.as_query_engine()


def main():
    st.title("Interactive Query Interface")

    # Model selector with a more subtle appearance
    selected_model = st.selectbox(
        "Model",  # Label is now visible
        options=["gpt-3.5-turbo", "gpt-4",
                 "gpt-4-turbo", "gpt-4o-mini", "gpt-4o"],
        index=0,
        label_visibility="visible"  # Changed from "collapsed" to "visible"
    )

    # Create a text input for the user's query
    user_query = st.text_input("Question:")

    if st.button("Submit Query"):
        if user_query:
            with st.spinner("Searching for an answer..."):
                answer = weaveindex.query(
                    user_query, query_engine, selected_model)

                st.write("Answer:")
                st.write(answer)
        else:
            st.warning("Please enter a query.")


if __name__ == "__main__":
    main()
