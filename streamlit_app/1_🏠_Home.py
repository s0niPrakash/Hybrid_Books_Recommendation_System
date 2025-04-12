import streamlit as st
import pandas as pd
import numpy as np 
from IPython.display import Image, display
import faiss
from scipy.sparse import csr_matrix , hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from implicit.als import AlternatingLeastSquares
import streamlit.components.v1 as components


st.set_page_config(
    page_title="Home"

)

def messages(errors):
    if errors == 'user_details_missed':
        st.info('ℹ️Personalize your experience! \n\nProviding your preferences helps this app work more efficiently for you. We truly appreciate it! 🙌')
        st.error('🚨 Missing details! \n\nPlease provide your details and preferences on the **👤User-details page**.')
    elif errors ==  'Required_files_missing':
        st.error('🚨 Some required files📂 are missing!. \n\nPlease go to **👤User-details page**. The system will update them automatically.')

@st.cache_data
def images_display(data,column):
    num_images=len(data)
    cols=st.columns(num_images)

    for i,(_,row) in enumerate(data.iterrows()):
        with cols[i]:
            st.image(row[column],use_container_width=True)



def images_displays(data, img_column, title_col, author_col, year_col, publisher_col):
    # Apply CSS for styling
    css = """
        <style>
        .image-box {
            display: flex;
            flex-wrap: nowrap;
            overflow-x: auto;
            gap: 15px;
            padding: 15px;
            border: 2px solid #ddd;
            border-radius: 10px;
            background-color: #f9f9f9;
            white-space: nowrap;
        }
        .image-box::-webkit-scrollbar {
            height: 8px;
        }
        .image-box::-webkit-scrollbar-thumb {
            background: #bbb;
            border-radius: 10px;
        }
        .image-box::-webkit-scrollbar-track {
            background: #f1f1f1;
        }
        .image-container {
            text-align: center;
            min-width: 160px;
            flex: 0 0 auto;
        }
        .image-container img {
            height: 180px;
            border-radius: 10px;
            cursor: pointer;
            display: block;
            margin: 0 auto;
            transition: transform 0.3s ease-in-out;
        }
        .image-container img:hover {
            transform: scale(1.1);
        }
        .caption {
            font-size: 14px;
            margin-top: 5px;
            color: #333;
            text-align: center;
            word-wrap: break-word;
            display: none;
            padding: 5px;
            border: 1px solid #ddd;
            border-radius: 5px;
            background-color: #fff;
        }
        .details-button {
            margin-top: 5px;
            padding: 5px 10px;
            font-size: 14px;
            border: 1px solid #ccc;
            background-color: transparent;
            color: black;
            border-radius: 5px;
            cursor: pointer;
            transition: background 0.3s;
        }
        .details-button:hover {
            background-color: #ddd;
        }
        </style>
        <script>
        function toggleDetails(id) {
            var details = document.getElementById(id);
            if (details.style.display === "none") {
                details.style.display = "block";
            } else {
                details.style.display = "none";
            }
        }
        </script>
    """

    images_html = '<div class="image-box">'
    
    for index, row in data.iterrows():
        img_url = row[img_column]
        title = row[title_col]
        author = row[author_col]
        year = row[year_col]
        publisher = row[publisher_col]
        details_id = f"details_{index}"

        
        images_html += f'''
            <div class="image-container">
                <a href="{img_url}" target="_blank">
                    <img src="{img_url}" alt="Book Image">
                </a>
                <button class="details-button" onclick="toggleDetails('{details_id}')">View Details</button>
                <div id="{details_id}" class="caption">
                    <strong>Title:</strong> {title}<br>
                    <strong>Author:</strong> {author}<br>
                    <strong>Year:</strong> {year}<br>
                    <strong>Publisher:</strong> {publisher}
                </div>
            </div>
        '''
    
    images_html += '</div>'
    
    full_html = f"{css}{images_html}"
    components.html(full_html, height=300, scrolling=True)




# we are cache the entire class instance in cache resource
@st.cache_resource
def popularity_based_recommender():
    return PopularityBasedRecommender()


# Popularity based recommendations 
class PopularityBasedRecommender:
    @st.cache_data
    def __init__(_self):
        _self.item_scores = None
        _self.ratings_data = None

    @st.cache_data
    def fit(_self,ratings_data, item_col, rating_col):
        _self.ratings_data=ratings_data
        item_popularity = ratings_data.groupby(item_col).agg(popularity_score=(item_col,'size'),
                                                             average_rating=(rating_col,'mean')).reset_index()
        
        sorted_item_popularity=item_popularity.sort_values(by=['popularity_score','average_rating'],ascending=[False,False])

        _self.item_scores = sorted_item_popularity
    
    @st.cache_data
    def recommend_popular_items(_self,top_n=20):
        if _self.item_scores is None:
            raise ValueError("Model has not been trained. Call 'fit' with rating_data.")
        return _self.item_scores.head(top_n)
    
    @st.cache_data
    def checks_isbn_isin(_self,original_books,top_num,item_col='ISBN'):
        valid_books=_self.item_scores[_self.item_scores[item_col].isin(original_books[item_col].unique())]
        popular_books=pd.merge(valid_books,original_books,on='ISBN',how='left')
        return popular_books.head(top_num)
    
if st.session_state.get('required_files') == 'All_required_files_are_present':
    if __name__ == "__main__":
        st.title('Hybrid Book Recommender 📚 ')
        st.subheader('✨ Must-Read Masterpieces')
        if st.button('📝info'):
            st.write(
                '➡️These are the most popular books, based on the average user ratings and the number of readers.'   
            )
        recommender=popularity_based_recommender()
        recommender.fit(ratings_data=st.session_state.Reindex_Ratings,item_col='ISBN',rating_col='Book-Rating')
        recommender.recommend_popular_items()
        popular_books=recommender.checks_isbn_isin(st.session_state.Books_modified,20)
        images_displays(popular_books,'Image-URL-L','Book-Title','Book-Author','Year-Of-Publication','Publisher')
else:
    messages('Required_files_missing')





# predifining the each tfidf vectorizer so that that we dont mix the words 
@st.cache_data
def TFIDFvectorizer(data):
    author_vectorizer = TfidfVectorizer(max_features=600, smooth_idf=True, sublinear_tf=True,ngram_range=(1, 2), min_df=2, max_df=0.9)
    publisher_vectorizer = TfidfVectorizer(max_features=394, smooth_idf=True, sublinear_tf=True,ngram_range=(1, 2), min_df=2, max_df=0.9)
    year_Of_publication_categorizes_vectorizer = TfidfVectorizer(max_features=10, smooth_idf=True, sublinear_tf=True, ngram_range=(1, 2), min_df=2, max_df=0.9)
    #fit the data with supperate tfidf vectorizer
    author_vectorizer.fit_transform(data['Book-Author'])
    publisher_vectorizer.fit_transform(data['Publisher'])
    year_Of_publication_categorizes_vectorizer.fit_transform(data['Year-Of-Publication-categorizes'])
    return author_vectorizer,publisher_vectorizer,year_Of_publication_categorizes_vectorizer

#content based recommendation system 
@st.cache_resource
def index_IVF_PQ(_tfidf_data):
    d=_tfidf_data.shape[1]
    m=20
    clusters=2048
    n_bits=8
    quantizer=faiss.IndexFlatL2(d)
    index=faiss.IndexIVFPQ(quantizer,d,clusters,m,n_bits)
    train_data=_tfidf_data.toarray().astype('float32')
    index.train(train_data)
    batch_size=1000
    for i in range(0,_tfidf_data.shape[0],batch_size):
        single_batch=(_tfidf_data[ i : i+batch_size].toarray().astype('float32'))
        index.add(single_batch)
    return index


# This function helps to search the index and return the similar books
def search_index(_user_query,_index,num_of_top_books,nprobe_value,books_info_data):
    query_vector=_user_query.toarray().astype('float32')
    query_vector = query_vector.reshape(1, -1)
    # checks the shapes
    assert query_vector.shape[1] == _index.d,f"Query vector dimension {query_vector.shape[1]} does not match FAISS _index {_index.d}"

    _index.nprobe=nprobe_value
    Distances, Indices =_index.search(query_vector,num_of_top_books)
    recommended_books=[books_info_data.iloc[idx] for idx in Indices.flatten() if 0<= idx <len(books_info_data)]
    Recommended_books=pd.DataFrame(recommended_books)
    return Recommended_books


def categeries_year(year):
    if year < 1900:
        return 'eighteen'
    elif year >=1900 and year <=1925:
        return 'nineteenone'
    elif year >1925 and year <=1950:
        return 'nineteentwo'
    elif year >1950 and year <=1975:
        return 'nineteenthree'
    elif year >1975 and year <2000:
        return 'nineteenfour'
    else:
        return 'twentiesone'
    

# This function will take the users requirements 

def user_specification(_author_vectorizer,_publisher_vectorizer,_year_Of_publication_categorizes_vectorizer):
    book_author=st.session_state.author_name
    publisher=st.session_state.publisher
    year_of_publication=st.session_state.year

    if year_of_publication:
        try:
            year_of_publication=int(year_of_publication)
            #function call 
            year_of_publication1=categeries_year(year_of_publication)
        except ValueError:
            print('Invalid input: please enter the year of publication in numbers')
            year_of_publication=None
    else:
        # If user doesn't want to give the year
        year_of_publication=None
    if book_author:
        author_matrix=_author_vectorizer.transform([book_author])
        print('Selected Author:',book_author)
    else:
        author_matrix=csr_matrix((1,author_vectorizer.get_feature_names_out().shape[0]))
        print('Selected Author: None')
    if publisher:
        publisher_matrix=_publisher_vectorizer.transform([publisher])
        print('Selected Publisher:',publisher)
    else:
        publisher_matrix=csr_matrix((1,publisher_vectorizer.get_feature_names_out().shape[0]))
        print('Selected Publisher: None')
    if year_of_publication:
        year_matrix=_year_Of_publication_categorizes_vectorizer.transform([year_of_publication1])
        print('Selected Year of Publication:',year_of_publication)
    else:
        year_matrix=csr_matrix((1,year_Of_publication_categorizes_vectorizer.get_feature_names_out().shape[0]))
        print('Selected Year of Publication: None')
    user_query_matrix=hstack([author_matrix,publisher_matrix,year_matrix])
    # Creating the dict 
    return user_query_matrix


# Collaberative Filter

# This function is to train the model with hyper parameters 
@st.cache_resource
def model_train(_non_sparse_matrix,factors,regularization,iteration):
    model=AlternatingLeastSquares(factors=factors,regularization=regularization,iterations=iteration)
    model.fit(_non_sparse_matrix.T)
    return model

def map_user_id_to_index(user_id, user_id_mapping):
    """Map a raw user ID to the matrix index."""
    return user_id_mapping.get(user_id, None)

def calculate_biases(actual_ratings,user_index):
    # Global biases
    global_bias=actual_ratings.sum() / actual_ratings.nnz # Number of non zero elements 
    
    # User biases 
    user_sum=actual_ratings[user_index, :].sum()
    user_nnz_count=(actual_ratings[user_index, :] != 0).sum()  # It create a boolen array and .sum() will count the true thats how we are going to know that how item does user rated 
    user_bias=(user_sum / user_nnz_count) - global_bias if user_nnz_count> 0 else 0

    # Item biases
    item_sum = actual_ratings[user_index, :].toarray().flatten()
    item_nnz_counts = (actual_ratings[user_index, :] != 0).toarray().flatten()
    item_bias = np.zeros(actual_ratings.shape[1])
    item_bias[item_nnz_counts > 0] = (item_sum[item_nnz_counts > 0] / item_nnz_counts[item_nnz_counts > 0]) - global_bias
    return global_bias ,user_bias, item_bias

def predict_ratings(user_factor,item_factors,global_bias,user_bias,item_bias):
    
    prediction_matrix = user_factor @ item_factors.T
    if user_bias==0 and np.all(item_bias == 0):
        return global_bias+prediction_matrix
    else:
        user_bias = np.array(user_bias).reshape(-1)
        item_bias = np.array(item_bias).flatten()
        # Align shapes by trimming item_bias if needed
        item_bias = item_bias[:prediction_matrix.shape[0]]
        return global_bias+user_bias+item_bias.flatten()+prediction_matrix 
    
def top_recommended_isbn(predictions_ratings,item_id_mapping,n=20):
    top_indices=np.argsort(predictions_ratings)[::-1][:n]
    top_recommends=predictions_ratings[top_indices]
    isbn_code = [(list(item_id_mapping.keys())[i], score) for i, score in zip(top_indices, top_recommends)]
    return isbn_code

def recommend_books(isbn_code,original_data,top_n):
    # creating a dataframe 
    isbn=pd.DataFrame(isbn_code,columns=['ISBN','Ratings'])
    # This will checks does we have the books details in our original dataset 
    valid_books=isbn[isbn['ISBN'].isin(original_data['ISBN'].unique())]
    # using left join we are going to retrive the data from the orinal dataset 
    top_recommends=pd.merge(valid_books,original_data,on='ISBN',how='left')
    return top_recommends.head(top_n)

if st.session_state.get('user_specifications_entered'):
    Books_modified_copy=st.session_state.Books_modified_copy
    Books_modified=st.session_state.Books_modified
    if st.session_state.get('user_id') is None:
        Content_based_filter_sparse=st.session_state.Content_based_filter_sparse
        # content based 
        if st.session_state.Books_modified_copy is not None:
            author_vectorizer,publisher_vectorizer,year_Of_publication_categorizes_vectorizer=TFIDFvectorizer(Books_modified_copy)
        if st.session_state.Content_based_filter_sparse is not None:
            Index=index_IVF_PQ(st.session_state.Content_based_filter_sparse)
        session_users_specifications=['year','publisher','author_name']
        if any(st.session_state.get(var) is not None for var in session_users_specifications):
            # i have to clear the years also 
            query=user_specification(author_vectorizer,publisher_vectorizer,year_Of_publication_categorizes_vectorizer)
            recommended_books = search_index(query,Index,5,30,st.session_state.Books_modified)
            st.subheader('🚀 Top 5 Recommended Books')
            if st.button('💡info'):
                st.write('➡️ These books are recommended based on the user preferences (Content-Based-Filtering), using indexIVFPQ (Inverted File Index With Product Quantization) from faiss (Facebook AI Similarity Search)')
            images_displays(recommended_books,'Image-URL-L','Book-Title','Book-Author','Year-Of-Publication','Publisher')

    else:
        # collabrative filter 
        model=model_train(st.session_state.Ratings_sparse_data,50,0.01,20)
        user_index=map_user_id_to_index(st.session_state.user_id,st.session_state.user_id_mapping)
        global_bias,user_bias,item_bias=calculate_biases(st.session_state.Ratings_sparse_data,user_index)
        user_factor=model.user_factors[user_index]
        item_factors=model.item_factors
        predictions_ratings=predict_ratings(user_factor,item_factors,global_bias,user_bias,item_bias)
        top_isbn=top_recommended_isbn(predictions_ratings,st.session_state.item_id_mapping)
        recommended_book=recommend_books(top_isbn,Books_modified,5)
        st.subheader('🔝 Top 5 Recommended Books')
        if st.button('🛈 info'):
            st.write('➡️ These books are recommended based on users past interaction(Collaborative-Filtering), using the Alternating Least Squares (ALS) algorithm.')
        images_displays(recommended_book,'Image-URL-L','Book-Title','Book-Author','Year-Of-Publication','Publisher')

else:
    messages('user_details_missed')