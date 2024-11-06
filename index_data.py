import weaveindex
import weave
import streamlit as st


docs_dir = 'crawled_docs'


def main():
    st.title("Weave Doc Indexer")

    # Initialize weave
    weave.init('weave-qa')

    docs_url = st.text_input(
        "Enter the documentation URL:", "https://github.com/wandb/weave/tree/master/docs/docs")

    col1, col2 = st.columns(2)

    with col1:
        if st.button("1. Crawl Docs"):
            with st.spinner("Crawling documentation..."):
                temp_docs = weaveindex.crawl_github_docs(docs_url, docs_dir)
                count_docs = len(temp_docs)

                dataset = weaveindex.load_files_as_dataset(docs_dir)
                ref = weave.publish(dataset, 'documents')

                if (count_docs > 0):
                    st.success(
                        f"{count_docs} docs crawled. Published to {ref.uri()}")
                    st.session_state['count_docs'] = count_docs
                else:
                    st.error("No docs found.")
    with col2:
        if st.button("2. Create Index"):
            if 'count_docs' not in st.session_state:
                st.error("Please crawl the docs first!")
            else:
                with st.spinner("Creating vector index..."):
                    vector_index = weaveindex.create_vector_index(
                        docs_dir)
                    st.success("Vector index created successfully!")
                    st.session_state['vector_index'] = vector_index


if __name__ == "__main__":
    main()
