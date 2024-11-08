import streamlit as st
from importlib import import_module
import pages.index_docs
import pages.generate_eval_data
import pages.evaluate
import pages.query_docs


def main():
    st.title("Document Intelligence Platform")


if __name__ == "__main__":
    main()
