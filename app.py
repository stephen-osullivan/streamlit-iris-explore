import streamlit as st
from sklearn.datasets import load_iris
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the iris dataset
iris = load_iris()

# Create a Streamlit app
st.title("Explore the Iris Dataset")
st.write("This is a simple app that allows the user to explore the Iris data set within Scikit-learn")

# Display the dataset info
st.subheader("Dataset Information")
st.write(f"The iris dataset contains {len(iris.data)} samples and {len(iris.feature_names)} features.")
st.write("The features are:")
for feature in iris.feature_names:
    st.write(f"- {feature}")
st.write(f"The target classes are: {', '.join(iris.target_names)}")

# Display the dataset
st.subheader("Dataset")
iris_df = pd.DataFrame(iris.data, columns=iris.feature_names)
iris_df['target'] = iris.target
st.dataframe(iris_df, use_container_width=True)

# Allow the user to select a feature to visualize
st.subheader("Visualize a Feature")
feature = st.selectbox("Select a feature to visualize:", iris.feature_names)
fig, ax = plt.subplots(figsize=(8, 6))
sns.histplot(data=iris_df, x=feature, hue='target', kde=True, ax=ax)
st.pyplot(fig)