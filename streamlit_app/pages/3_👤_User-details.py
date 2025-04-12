import pandas as pd
pd.set_option('display.max_columns', 15)
import warnings
warnings.filterwarnings('ignore')
import numpy as np
from scipy.sparse import load_npz
import pickle
import streamlit as st
import os
import time

def load_user_mapping(filename):
    with open(filename, 'rb') as f:
        return pickle.load(f)

# Create a placeholder for the message
message_placeholder = st.empty()

current_dir=os.getcwd()
message_placeholder.info(f'ℹ️ Checking for required files in {current_dir}')
time.sleep(0.5)
required_files = [
    'Content_based_filter_sparse.npz',
    'ratings_sparse_matrix.npz',
    'user_id_mapping.pkl',
    'item_id_mapping.pkl',
    'rating_modified.pkl',
    'books_modified_original.pkl',
    'data_for_trending.pkl'
]

missing_files=[file for file in required_files if not os.path.exists(os.path.join(current_dir, file))]
if missing_files:
    st.error('❌ Some required files are missing!')
    st.write('Please make sure to place the following files in the same directory as this Streamlit app:')
    for file in missing_files:
        st.write(f'- {file}')

else:
    st.session_state.Content_based_filter_sparse=load_npz('Content_based_filter_sparse.npz')
    st.session_state.Books_modified_copy=load_user_mapping('books_no_stopwords_modified.pkl')
    st.session_state.Ratings_sparse_data=load_npz('ratings_sparse_matrix.npz')
    st.session_state.user_id_mapping=load_user_mapping('user_id_mapping.pkl')
    st.session_state.item_id_mapping=load_user_mapping('item_id_mapping.pkl')
    st.session_state.Reindex_Ratings=load_user_mapping('rating_modified.pkl')
    st.session_state.Books_modified=load_user_mapping('books_modified_original.pkl')
    st.session_state.data_for_trending=load_user_mapping('data_for_trending.pkl')
    st.session_state.required_files = 'All_required_files_are_present'
    message_placeholder.success('✅ All required files are present!')
    Reindex_Ratings=st.session_state.Reindex_Ratings
    Books_modified=st.session_state.Books_modified
    time.sleep(0.5)
message_placeholder.empty()

@st.cache_data
def interaction_users(data):
    # user with books interactions 
    user_id_with_interaction=set(data[data['Book-Rating']>0]['User-ID'].unique())
    user_id_list = ['None'] + sorted([int(uid) for uid in user_id_with_interaction])
    return user_id_list
 
@st.cache_data
def book_details(data):
    author_unique_names=data['Book-Author'].unique()
    author_names_list = ['None'] + [author for author in author_unique_names]

    year_of_pulication_unique=data['Year-Of-Publication'].unique()
    year_of_publicaton_list=['None'] + sorted([int(year) for year in year_of_pulication_unique])

    publisher_names_unique=data['Publisher'].unique()
    publisher_names_list=['None'] + [publisher for publisher in publisher_names_unique]

    books_names_unique=data['Book-Title'].unique()
    books_title_list=[books for books in books_names_unique]


    return author_names_list,year_of_publicaton_list,publisher_names_list,books_title_list


if Reindex_Ratings is not None:
    user_id_list = interaction_users(Reindex_Ratings)
if Books_modified is not None:
    author_names,year_of_publications,publisher_names,book_titles=book_details(Books_modified)
    st.session_state.book_titles=book_titles

def initialize_session_state():
    # Initialize session state variables if not already set
    if 'user_specifications_entered' not in st.session_state:
        st.session_state.user_specifications_entered = False
    if 'user_id' not in st.session_state:
        st.session_state.user_id = None
    if 'author_name' not in st.session_state:
        st.session_state.author_name = None
    if 'year' not in st.session_state:
        st.session_state.year = None
    if 'publisher' not in st.session_state:
        st.session_state.publisher = None
    if 'user_details' not in st.session_state:
        st.session_state.user_details=None

initialize_session_state()

# Function to reset user details
def reset_details():
    st.session_state.user_specifications_entered = False
    st.session_state.user_id = None
    st.session_state.author_name = None
    st.session_state.year = None
    st.session_state.publisher = None
    st.rerun()  # Refresh the app

# **Page 1: User Details**
if not st.session_state.user_specifications_entered:
    st.title('📝 Please Enter Your Interests')
    st.info('ℹ️ You can either **select a User ID** or **keep it a None**.')
    st.warning('**NOTE:** If you do selected a User ID then you dont have to give your preferences.')
    st.info('ℹ️ If you didnt selected a User ID,**some features may not work**. Please provide your preferences')

    # creating a form that will take all the values at a time 
    with st.form('my_form'):
        selected_user_id = st.selectbox('Select User ID',options=user_id_list)
        selected_author_name = st.selectbox('Select Author Name:',options=author_names)
        selected_year = st.selectbox('Enter Year of Publication:',options=year_of_publications)
        selected_publisher = st.selectbox('Enter Publisher Name:',options=publisher_names)
        my_picks=st.form_submit_button('Submit my picks----')
    
    if my_picks is not None:
        selected_user_id= None if selected_user_id == 'None' else selected_user_id
        selected_author_name= None if selected_author_name == 'None'else selected_author_name
        selected_year=None if selected_year == 'None'else selected_year
        selected_publisher=None if selected_publisher == 'None'else selected_publisher

        st.session_state.user_id = selected_user_id
        st.session_state.author_name = selected_author_name
        st.session_state.year = selected_year
        st.session_state.publisher = selected_publisher
    if all( value is None for value in [selected_user_id, selected_author_name, selected_year, selected_publisher]):
        st.warning('⚠️ Please fill at least one box.')
        st.toast('Please fill at least one box.', icon="⚠️")
    else:
        st.session_state.user_details = 'user_details_are_entered'
        st.session_state.user_specifications_entered = True
        st.rerun()

else:
    st.title('📌 User Details & Preferences ')
    st.write(f'👤 **User ID:** {st.session_state.user_id}')
    st.write(f'✍️ **Author Name:** {st.session_state.author_name}')
    st.write(f'📅 **Year of Publication:** {st.session_state.year}')
    st.write(f'🏢 **Publisher Name:** {st.session_state.publisher}')
    st.toast('your preference have been saved!', icon="👍")

    # Logout button to reset details
    if st.button('Logout'):
        reset_details()
