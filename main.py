import streamlit as st
from importlib import import_module
import pages.index_docs
import pages.generate_eval_data
import pages.evaluate
import pages.query_docs


def main():

    st.title("Weave Powered Document Indexing, Querying and Evaluation")


if __name__ == "__main__":
    main()
