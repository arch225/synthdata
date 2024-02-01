import streamlit as st
import SessionState

import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

https://github.com/codebasics/py/blob/master/ML/17_knn_classification/knn_classification_tutorial.ipynb
import pandas as pd

# Sample DataFrame with a categorical variable
data = {'Category': ['A', 'B', 'C', 'A', 'C', 'B']}
df = pd.DataFrame(data)

# One-hot encoding using get_dummies
one_hot_encoded = pd.get_dummies(df['Category'], prefix='Category')

# Concatenate the one-hot encoded columns to the original DataFrame
df_encoded = pd.concat([df, one_hot_encoded], axis=1)

# Display the updated DataFrame with one-hot encoding
print(df_encoded)


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
