# import streamlit as st
# st.title("Hello World")

import weave


client = weave.init("wandb/hellaswag")
call = client.get_call("f976bab6-f5c3-4c18-bb6c-db396d98b8fd")
breakpoint()

print(call)
weave.Object.WeaveCall 