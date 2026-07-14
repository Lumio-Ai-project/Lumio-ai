import streamlit as st
from workflow import WorkFlow
from langchain_core.messages import HumanMessage, SystemMessage
from models import State

if "messages" not in st.session_state:
    st.session_state.messages = []

st.title("New Article Generation")

for msg in st.session_state.messages:
    with st.chat_message(msg['role']):
        st.markdown(msg['content'])

if prompt := st.chat_input("Ask any question..."):
    st.session_state.messages.append({"role": "user", "content": prompt})

    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        chat_history = [HumanMessage(content=msg['content']) for msg in st.session_state.messages]

        init_state = State(
            chat_history=chat_history,
            query=prompt,
            context=None,
            response="",
            query_rewrite=""
        )

        result = WorkFlow().run(init_state)
        response = result.get("response")
        st.markdown(response)
        st.session_state.messages.append({"role": "assistant", "content": response})