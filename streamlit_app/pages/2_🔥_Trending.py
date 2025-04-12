import streamlit as st
import pandas as pd
import numpy as np 
import streamlit.components.v1 as components

st.set_page_config(page_title='Trending')

def messages(errors):
    if errors == 'user_details_missed':
        st.info('ℹ️Personalize your experience! \n\nProviding your preferences helps this app work more efficiently for you. We truly appreciate it! 🙌')
        st.error('🚨 Missing details! \n\nPlease provide your details and preferences on the **👤User-details page**.')
    elif errors ==  'Required_files_missing':
        st.error('🚨 Some required files📂 are missing!. \n\nPlease go to **👤User-details page**. The system will update them automatically.')


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

        if not isinstance(img_url, str) or not img_url.startswith("http"):
            continue  # Skip invalid image URLs
        
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




@st.cache_resource
def trending_books_filter():
    return Trending_books()

class Trending_books:
    def __init__(self):
        self.dataset = None
        self.treding_items = None
    def fit(self,dataset):
        self.dataset=dataset
    def pick_trending_books(self,year,data_col,item_col,rating_col,top_n):
        if self.dataset is None:
            raise ValueError("Model has not been trained. Call 'fit' with interaction data.")
        year_data=self.dataset[self.dataset[data_col] == year]
        if year_data is None:
            raise ValueError(f"we dont have the books details of {year} year ")
        item_trending=year_data.groupby(item_col).agg(popularity_score=(item_col,'size'),
                                                       average_rating=(rating_col,'mean')).reset_index()
        trending_items=item_trending.sort_values(by=['popularity_score','average_rating'], ascending=[False, False])
        self.trending_items = trending_items.head(top_n)
    def book_recommender(self,original_data):
        book_details=pd.merge(self.trending_items,original_data,on="ISBN",how='left')
        return book_details



if st.session_state.get('required_files') == 'All_required_files_are_present':
    st.title('🔥Trending Books')
    st.info("✨**Trending Books of the Year!**\n\n Explore top-rated books based on popularity and ratings. ")

    if __name__ == "__main__":
        year_of_pulication_unique=st.session_state.Books_modified['Year-Of-Publication'].unique()
        year_of_publicaton_list=sorted([int(year) for year in year_of_pulication_unique])
        no_of_books_list=[5,10,15]
        with st.form('my-form'):
            st.session_state.selected_YEAR = st.selectbox('Enter year:',options=year_of_publicaton_list,index=year_of_publicaton_list.index(2005))
            st.session_state.no_of_books=st.selectbox('Enter number of books to show:',options=no_of_books_list,index=no_of_books_list.index(5))
            picks=st.form_submit_button('Submit my picks----')
        trend_recommender=trending_books_filter()
        trend_recommender.fit(st.session_state.data_for_trending)
        trend_recommender.pick_trending_books(st.session_state.selected_YEAR,'Year-Of-Publication','ISBN','Book-Rating',st.session_state.no_of_books)
        trending_books=trend_recommender.book_recommender(st.session_state.Books_modified)
        st.markdown("""<div style="padding: 10px; border-radius: 8px; background-color: #e8f5e9; 
            border-left: 4px solid #2e7d32; font-size: 16px; width: 60%;">
                    <b>📖 Here are the trending books of the year!</b></div>""", unsafe_allow_html=True)
        images_displays(trending_books,'Image-URL-L','Book-Title','Book-Author','Year-Of-Publication','Publisher')
else:
    messages('Required_files_missing')

