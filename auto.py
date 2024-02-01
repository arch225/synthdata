import streamlit as st
import SessionState

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Sample DataFrame
data = {'value': [10, 15, 20, 25, 30, 35],
        'indicator': ['A', 'B', 'A', 'B', 'A', 'B']}

df = pd.DataFrame(data)

# Create subplots with histograms based on the 'indicator' column
g = sns.FacetGrid(df, col='indicator', height=4, sharey=False)
g.map(plt.hist, 'value', bins=10, color='skyblue')

# Set plot labels and titles
g.set_axis_labels('Value', 'Frequency')
g.set_titles('Histogram for Indicator {col_name}')

# Show the plots
plt.show()

def main():
    st.subheader("new")

    session_state = SessionState.get(name="", button_sent=False)

    session_state.name = st.text_input("Enter your name")
    button_sent = st.button("Send")

    if button_sent:
        session_state.button_sent = True

    if session_state.button_sent:
        st.write(session_state.name)

        session_state.bye = st.checkbox("bye")
        session_state.welcome = st.checkbox("welcome")

        if session_state.bye:
            st.write("I see")
        if session_state.welcome:
            st.write("you see")


main()
