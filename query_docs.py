import weaveindex
import weave
import streamlit as st

docs_dir = weaveindex.temp_dir


vector_index = weaveindex.load_vector_index()


# # query index
# query = "What is Weave?"
# query_engine = vector_index.as_query_engine()
# response = query_engine.query(query)

# print(response)

# Add the following Streamlit interface code at the end of the file
def main():
    st.title("Weave Documentation Query Interface")

    # Load the vector index
    vector_index = weaveindex.load_vector_index()
    query_engine = vector_index.as_query_engine()

    # Create a text input for the user's query
    user_query = st.text_input("Enter your question about Weave:")

    if st.button("Submit Query"):
        if user_query:
            with st.spinner("Searching for an answer..."):
                answer = weaveindex.query(user_query, query_engine)

                st.write("Answer:")
            st.write(answer)
        else:
            st.warning("Please enter a query.")


if __name__ == "__main__":
    main()
