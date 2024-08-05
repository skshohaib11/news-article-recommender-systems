import streamlit as st
import pandas as pd
import numpy as np
import pickle
import zipfile
import sklearn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import euclidean_distances
from sklearn.metrics.pairwise import pairwise_distances

  

# Load the saved model
with open(r'model_tfidf.pkl', 'rb') as f:
    model = pickle.load(f)

# Load the data
# Declare global variable
data = None
file_path = r'user.zip.pkl'
def unzip_csv_file(file_path):
    global data  # Declare the variable as global
    with zipfile.ZipFile(file_path, 'r') as zip_ref:
        csv_file = 'user_data.csv'  # Assuming the CSV file name is user_data.csv
        with zip_ref.open(csv_file) as file:
            data = pd.read_csv(file)  # Assign to the global variable

# Streamlit app code
def main():
    global data  # Declare the variable as global
    st.title("News Recommendation System")
    st.write("Please Check The Sidebar By clicking On Top Left Bar Arrow.")
    # Adding a sidebar title
    st.sidebar.title("How Does This Work?")

    # Add sidebar options
    st.sidebar.write("* There Are Total 147706 Users In Our Data. Follow the steps below: ")
    st.sidebar.write("* Step 1 - You Have To Load The Zip File Containing The Users Data You Can Find the File In My Github link below i.e., user.zip. If the File Is Already Uploaded We Can Skip Step 1.")
    st.sidebar.write("* Step 2 - You Have To Select A Particular User And Number Of Recommendations Needed For That Particular User.")
    st.sidebar.write("* Step 3 - You Can Select From The Dropdown List Or You can Just Type User_ID In The Format for example user_14401")
    st.sidebar.write("* Step 4 - Hit The Get Recommendations Button To Get The Recommendations.")


    st.sidebar.write("Prepared By: Shohaib Shaikh")
    st.sidebar.markdown("[Github](https://github.com/skshohaib11/Data_Detectives)-----[linkedIn](https://www.linkedin.com/in/shaikh-shohaib-043317251/)")

    # File uploader
    file = st.file_uploader("Upload a ZIP file", type="zip")

    if file is not None:
        # Unzip and read CSV file
        unzip_csv_file(file)

        # Button to display DataFrame
        if st.button("Display DataFrame"):
            if data is not None:
                st.dataframe(data)

if __name__ == '__main__':
    main()


# Function to calculate Euclidean similarity
def euclidean_similarity(tfidf_matrix, user_index):
    user_vector = tfidf_matrix[user_index]
    distances = pairwise_distances(tfidf_matrix, user_vector)
    return 1 / (1 + distances.flatten())

# Function to get recommendations
def get_recommendations(user_id, num_recommendations):
    # Convert user_id to integer index
    user_index = int(user_id.split('_')[1]) - 1
    
    # Calculate the TF-IDF matrix
    tfidf_vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = tfidf_vectorizer.fit_transform(data['short_description'])
    
    # Calculate the Euclidean similarity scores incrementally
    similarity_scores = euclidean_similarity(tfidf_matrix, user_index)
    
    # Sort the similarity scores in descending order
    sorted_indices = np.argsort(similarity_scores)[::-1]
    
    # Get the top num_recommendations indices (excluding the given user_index)
    top_indices = sorted_indices[1:num_recommendations+1]
    
    # Get the corresponding recommendations from the data
    recommendations = data.iloc[top_indices][['category', 'headline', 'link', 'short_description']]
    
    return recommendations

# Set up the UI
st.write("Get Upto 10 Best Recommendations for Your User.")


# Set up the user ID input
user_id = st.selectbox("Select a user ID", data['user_id'])

# Set up the number of recommendations input
num_recommendations = st.number_input("Number of Recommendations", min_value=1, max_value=10, value=5)

# Set up the recommendations button
if st.button("Get Recommendations"):
    recommendations = get_recommendations(user_id, num_recommendations)
    recommendations = recommendations.reset_index(drop=True)  # Reset the index
    st.write("Here are your recommendations:")
    st.table(recommendations)
 
