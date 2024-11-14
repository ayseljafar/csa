import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import StringIO
from joblib import load
from sklearn.feature_extraction.text import TfidfVectorizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
import re
from nltk.stem import WordNetLemmatizer
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')
nltk.download('wordnet')
nltk.download('omw-1.4')

# Read the dataset
df = pd.read_csv("reviews.csv", lineterminator='\n', header=0)
# Assign unique index numbers
df.reset_index(drop=True, inplace=True)


# Set page configuration
st.set_page_config(
    page_title="CUSTOMER SATISFACTION",
    page_icon="🌟",
    layout="wide")
    


# Title and Image for Introduction
st.title("CUSTOMER SATISFACTION | Trustpilot reviews")


# Sidebar with Table of Contents
st.sidebar.title("Table of Contents")
pages = ["Introduction", "Data Exploration", "Data Visualization", "Machine Learning Models",
         "Cluster Analysis", "Neural Network Model", "Conclusion", "Sentiment Prediction"]
page = st.sidebar.radio("Go to", pages)


# Introduction Page
if page == pages[0]:
    # Display a title
    st.header('Behind the Stars: The Art and Science of Trustpilot Sentiment Analysis')
    
   # Use columns to display the image and text side by side
    col1, col2 = st.columns([2,1])
    
    # Display the project description text in the second column
    # Project Description
    col2.write(' ')
    col2.write(' ')
    col2.write(' ')
    col2.write(' ')
    col2.write(' ')
    col2.write(' ')
    col2.write(' ')
    col2.write(' ')
    col2.write(' ')
    
    col2.markdown(
        """
        ## Project Description:
        
        In the dynamic landscape of the digital era, the influence of online reviews on consumer behavior cannot be overstated.
        Platforms like Trustpilot have become pivotal in shaping customer decisions and, consequently, the success of businesses.
        Understanding the sentiments expressed in customer reviews is not just an art; it's a science.
        This project delves into the intricacies of Trustpilot sentiment analysis, unraveling the nuances of customer feedback to provide actionable insights. 
        
        """)
    
    # Display the introduction image in the first column
    col1.image("images/Introduction.jpg", use_column_width=True)
    # Add a centered caption to the image with an HTML link
    caption_text = 'Image by <a href="https://de.freepik.com/vektoren-kostenlos/winziger-analytiker-beobachtet-die-leistung-der-arbeiter-auf-dem-tablet-leistungsbewertung-messung-der-mitarbeiterarbeit-feedback-konzept-zur-arbeitseffizienz_10782800.htm#page=6&query=online%20reviews%20handy&position=19&from_view=search&track=ais&uuid=ba32c2f7-5357-4228-af61-9e5663710a4d" target="_blank">vectorjuice on Freepik</a>'
    col1.markdown(f'<div style="text-align:center">{caption_text}</div>', unsafe_allow_html=True)
   
    
    
    
    st.write(' ')
    st.write(' ')
    # Project Description
    st.markdown(
        """
        ## Dataset:
        
        Our exploration begins with a dataset collected by web scraping Trustpilot.com using Python's BeautifulSoup library.
        Trustpilot is a renowned online review platform celebrated for its extensive collection of user-generated reviews. 
        Focusing specifically on the appliance and electronics categories, this dataset offers a treasure trove of information to decipher customer sentiments. 
        From positive accolades to critical feedback, the dataset captures the diverse spectrum of opinions, painting a comprehensive picture of customer experiences.
        
        """)
    
    
    st.write(' ')
    st.write(' ')
    # Project Goals
    st.markdown(
        """
        ## Project Goals:

        - **Effective Data Cleaning and Preparation:** Ensuring our data is pristine and ready for analysis.
        - **Accurate Sentiment Classification:** Developing models to precisely classify sentiments in customer reviews.
        - **High Accuracy and Performance:** Achieving reliable and high-performing machine learning models.
        - **Effective Visualization and Interpretation:** Providing clear insights through visualizations and model interpretations.
        - **Customer Satisfaction and Engagement:** Understanding and improving customer satisfaction through feedback analysis.
        """)
    
    st.write(' ')
    st.write(' ')
    st.markdown(
        """
       In achieving these objectives, our project seeks to empower businesses with actionable recommendations and valuable information, 
       elevating their understanding of customer sentiments and guiding strategic decision-making. Join us on this journey behind the stars, 
       where the art and science of Trustpilot sentiment analysis come together to shape a customer-centric future.
        """)
    
    

     
# Data Exploration Page
elif page == pages[1]:
     
    st.write("## Presentation of the original dataset")
    st.dataframe(df.head(5))
    if  st.checkbox("Show Information of the original dataset") :
      
        # Use columns to display the info and description side by side
        col1, col2 = st.columns([2,2])

        # Display the introduction image in the first column
        with col1:
            st.subheader('DataFrame Info')
        
            # Use a StringIO object to capture the output of df.info()
            buffer = StringIO()
            df.info(buf=buffer)
            s = buffer.getvalue()
            # Display the captured output in Streamlit
            st.text(s)

        # Display column descriptions in the second column
        with col2:
            st.subheader('Column Descriptions')
            st.markdown('''
                        - Name: Username of the customer who wrote the comment
                        - Number of reviews: Number of reviews the customer has written
                        - Country: Country of the customer
                        - Title: Title of the comment
                        - Comment: Main body of the customer's comment
                        - Date of comment: Date when the customer's comment was posted
                        - Date of experience: Date of the customer's experience with the product
                        - Company: Company about which the comment was written
                        - Response: Captures any response provided by the company
                        - Date of response: Date of the company's response
                        - Stars: Rating given by the customer
                        ''')
    
    st.text(" ")
    st.text(' ')          
    # Use columns to display the image and text side by side
    col1, col2, = st.columns([1,1])
    
    # Display the project description text in the second column
    col1.text(' ') 
    col1.text(' ') 
    col1.markdown(
        """
        ### Preparation of the dataset:

        - Removing missing values
        - Adding a new column 'Sentiment"
        - Converting strings into numeric variables
        - Setting 'Response' to 1 if there is a response from the company and 0 if there is no response from the company
        - Deleting unusable Columns
        - Adding 'Title' to 'Comment'
        - Splitting the dates into days, months and years
        - Feature Engineering from Comments
        - Text cleaning
        """)   
    # Display the introduction image in the first column
    col2.image("images/Preparation.jpg", use_column_width=True) 
    # Add a centered caption to the image with an HTML link
    caption_text = 'Image by <a href="https://de.freepik.com/vektoren-kostenlos/datenwissenschaftler-datenanalysemanager-datenbankentwickler-und-administrator-arbeiten-big-data-job-datenbankentwickler-karrieren-im-big-data-konzept-helle-lebendige-violette-isolierte-illustration_10780537.htm#query=Preparation%20Dataset&position=48&from_view=search&track=ais&uuid=f171ea0e-0a3b-45b3-9f36-353207576000" target="_blank">vectorjuice on Freepik</a>'
    col2.markdown(f'<div style="text-align:center">{caption_text}</div>', unsafe_allow_html=True)     
         
    st.text('')
    st.text('')
    st.write("## Example for Text cleaning")
    # Generate stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update([",", ".", "?", ":", "!", "’", "/", ")", "'s", "(", "hair", "n't"])

    # Function to filter stopwords
    def stop_words_filtering(text):
        text = str(text).lower()
        tokens = text.split() 
        tokens_filtered = [word for word in tokens if word not in stop_words]
        return " ".join(tokens_filtered)

    # Function to clean the text
    def review_cleaning(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[^\w\s]', '', text)
        return text

        
    # Input field for text
    user_input = st.text_area("Enter your Text:", "", height= 250)
    
    # Apply text cleaning functions
    cleaned_text = user_input
    
    # Checkbox to apply text cleaning
    if st.checkbox("Clean your text"): 
            
            cleaned_text = stop_words_filtering(cleaned_text)
            cleaned_text = review_cleaning(cleaned_text)
            # Output after Text cleaning
            st.markdown(' ') 
            st.markdown(' **Text after text cleaning:** ') 
            st.text(cleaned_text)
            
    
    st.write("## Presentation of the prepared dataset")
    # Load the dataset
    df_prepared = load('joblib/df.pkl')
    
    st.dataframe(df_prepared.head(5))
    if  st.checkbox("Show Information of the prepared dataset") :
      
        # Use columns to display the info and description side by side
        col1, col2 = st.columns([2,2])

        # Display the introduction image in the first column
        with col1:
            st.subheader('DataFrame Info')
        
            # Use a StringIO object to capture the output of df.info()
            buffer = StringIO()
            df_prepared.info(buf=buffer)
            s = buffer.getvalue()
            # Display the captured output in Streamlit
            st.text(s)

        # Display column descriptions in the second column
        with col2:
            st.subheader('Column Descriptions')
            st.markdown('''
                        - Sentment: 0 for negative rating, 1 for positive rating
                        - Number of reviews: Number of reviews the customer has written
                        - Comment: Main body of the customer's comment
                        - Response: Captures any response provided by the company
                        - com_day, com_month, com_year: Date when the customer's comment was posted
                        - res_day, res_month, res_year: Date of the company's response (No response= 0)
                        - Num_of_words: Number of words in each comment
                        - Capital_count: Number of capital letters in each comment 
                        - Smile_count: Count of positive emojis in a comment
                        - Sad_count: Count of negative emojis in a comment
                        - Exclamation_count: Number of exclamation marks used in a comment, 
                        - Question_count: Number of question marks in a comment
                        - Integer_count: Number of digits present in the comments
                        ''')
    
     
    
   
           
elif page == pages[2] : 
    st.write("### Visualization of data")
    
   # Select a Visualisation
    choice = ["Distribution of the Sentiment", "Distribution of the Responses", "Features from comments", "Wordcloud",
             "Most used words", "Heatmap"]
    option = st.selectbox("Select a Visualization", choice) 
    
    if option == choice[0]:
        # Sentiment Distribution Bar Chart
        st.subheader("Sentiment Distribution based on Star Ratings")
        st.write("The bottom bar chart categorizes sentiments into “neg” (negative reviews) and “pos” (positive reviews) "
                "and provides insight into the sentiment distribution based on the star ratings.")
        
        # Use columns to display the description and image side by side
        col1, col2 = st.columns([2,2])

         # Display the image in the first column
        with col1:   
            st.image("images/Distribution of the Sentiment.png", use_column_width=True)  

        # Display the description in the second column
        with col2:
            
            # Analysiere und interpretiere die Sentiment-Verteilung
            neg_percentage = 21.65
            pos_percentage = 78.35

            # Sentiment Analysis Results
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write("- **Negative Sentiments ('neg'):** Approximately 21.65% of reviews carry a negative sentiment, "
                    "corresponding to star ratings 1, 2, and 3. This implies that roughly one-fifth of the reviews were classified as negative.")

            st.write("- **Positive Sentiments ('pos'):** Around 78.35% of reviews are characterized by a positive sentiment. "
                     "This indicates that the vast majority of reviews fall into the positive sentiment category.")

            # Interpretation
            st.write("- **Interpretation:** It is evident from the data that there is an imbalance between sentiment categories. Positive sentiments ('pos') "
                    "occur significantly more frequently than negative sentiments ('neg'). During later stages of machine learning, "
                    "it should be ensured that the model effectively learns from both positive and negative sentiments.")
         
       
        
         
        
        
    elif option == choice[1]:
        # Responses to Reviews
        st.subheader("Responses to Reviews")
        st.write("In the detailed examination of sentiments and responses, the distribution of sentiments ('neg' for negative reviews "
                "and 'pos' for positive reviews) in relation to responses ('No Response' and 'Response') is explored:")
        
        # Use columns to display the description and image side by side
        col1, col2 = st.columns([2,2])

         # Display the image in the first column
        with col1:   
            st.image("images/Response.png", use_column_width=True)  

        # Display the description in the second column
        with col2:
            
            # Negative Reviews and Responses
            st.write("**Negative Reviews:**")
            st.write("- Approximately 9.71% of negative reviews did not receive any response to their comments.")
            st.write("- Conversely, almost 12% of negative reviews received a reply.")
            st.write("- This suggests that companies actively engage with critical feedback, demonstrating a commitment to addressing potential issues or alleviating concerns.")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            # Positive Reviews and Responses
            st.write("**Positive Reviews:**")
            st.write("- For positive reviews, an overwhelming 70.87% received no response from the respective companies.")
            st.write("- Only 7.48% of positive reviews received a reply.")
            st.write("- It appears that companies may allocate fewer resources to engaging with already satisfied customers, potentially focusing more on addressing and improving negative experiences.")

        # Interpretation
        st.write("The observed pattern, with a high number of positive reviews and a low response rate, prompts consideration for companies "
                "to invest more resources in engaging with satisfied customers. Proactive engagement with positive feedback can contribute "
                "to customer retention and further enhance the positive image of the companies. This balanced approach to responding to both "
                "positive and negative reviews can contribute to a more comprehensive and customer-centric engagement strategy.")
         
        
        
    elif option == choice[2]:
        st.title('Feature Engineering from the comments')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        
        # Use columns to display the description and image side by side
        col1, col2, col3, col4 = st.columns([1,2,1,2])

        # Display the image in the first column
        with col1:   
            st.image("images/Num_of_words.png", use_column_width=True)  

        # Display the description in the second column
        with col2:
            st.write(' ')
            st.subheader("**Number of words**")
            st.write(' ')
            st.write('-> negative comments have a higher number of words ')
            
        # Display the image in the third column
        with col3:   
            st.image("images/Capital_count.png", use_column_width=True)  

        # Display the description in the fourth column
        with col4:
            st.write(' ')
            st.subheader("**Number of capital letters**")
            st.write(' ')
            st.write('-> negative comments have a higher number of capital letters ')
        
        # Use columns to display the description and image side by side
        col5, col6, col7, col8 = st.columns([1,2,1,2])

        # Display the image in the first column
        with col5:   
            st.image("images/Smile_count.png", use_column_width=True)   

        # Display the description in the second column
        with col6:
            st.write(' ')
            st.subheader("**Number of positive emojis**")
            st.write(' ')
            st.write('-> positive comments have a higher number of positive emojis ') 
            
        # Display the image in the third column
        with col7:   
            st.image("images/Sad_count.png", use_column_width=True)  

        # Display the description in the fourth column
        with col8:
            st.write(' ')
            st.subheader("**Number of negative emojis**")
            st.write(' ')
            st.write('-> negative comments have a higher number of negative emojis ')
        
        # Use columns to display the description and image side by side
        col9, col10, col11, col12 = st.columns([1,2,1,2])
        
        # Display the image in the first column
        with col9:   
            st.image("images/Question_count.png", use_column_width=True)  

        # Display the description in the second column
        with col10:
            st.write(' ')
            st.subheader("**Number of question marks**")
            st.write(' ')
            st.write('-> negative comments have a higher average number of question marks ')

        # Display the image in the first column
        with col11:   
            st.image("images/Exclamation_count.png", use_column_width=True)
              
        # Display the description in the second column
        with col12:
            st.write(' ')
            st.subheader("**Number of exclamation marks**")
            st.write(' ')
            st.write('-> negative comments have a higher average number of exclamation marks ')
            
        
    elif option == choice[3]:
        
        # Use columns to display the description and image side by side
        col1, col2, col3 = st.columns([2,1,2])

        # Display the image in the first column
        with col1:
            st.subheader("**Wordcloud pos**")
            st.image("images/Wordcloud pos.png", use_column_width=True)
            st.write(' ')
            st.markdown(''' 
                - Reflect a positive sentiment (love, happy)
                - Satisfaction with the product (great, product)
                - Fast Delivery (quick) ''')

        # Display the description in the second column
        with col3:
            st.subheader("**Wordcloud neg**")
            st.image("images/Wordcloud neg.png", use_column_width=True)
            st.write(' ')
            st.markdown(''' 
                    - Problems with the product (problem, product)
                    - Problems with the ordering process (order, time)
                    - Problems with the website (website) ''')

        st.write(' ')
        st.write(' ')
        
        st.markdown('### "Customer service" and "product" play a central role in the data set, for both positive and negative sentiment!')  
         
      
    elif option == choice[4]:
        
        # Most used words
        st.title("")
        images = ["Unigrams", "Bigrams", "Trigrams"]
        image = st.radio(" ", images)
        
        if image == images[0]:
            st.subheader(' ')
            st.image("images/Monograms.png", use_column_width=True)  
        elif image == images[1]:
            st.subheader(' ')
            st.image("images/Bigrams.png", use_column_width=True)
        elif image == images[2]:
            st.subheader(' ') 
            st.image("images/Trigrams.png", use_column_width=True)  
     
     
    elif option == choice[5]:
        
        st.write(' ')
        # Use columns to display the description and image side by side
        col1, col2 = st.columns([1,2])

       
        with col1:
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write('   **Positive correlation:**')
            st.write("   - between all variables related to firms' response to reviews")
            st.write('   - between Number of words and Number of digits in a comment')
            st.write('   - between Number of words and Number of capital letters in a comment')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write('    **Negative correlation:**')
            st.write("    - between Sentiment and variables related to firms' response")
            st.write('    - between Sentiment and Number of words in a comment')
            st.write('    - between Sentiment and Number of digits in a comment')
        
        with col2:
            st.image("images/Heatmap.png", use_column_width=True)  
            
  

elif page == pages[3] : 
    st.write("### Machine Learning models")
   
    
    # Select a Model
    choice1 = ["Models for metadata", "Models for textdata"]
    option1 = st.selectbox("Select data", choice1)
    
    if option1 == choice1[0]:
        # Use columns to display the results and image side by side
        col1, col2 = st.columns([1,1])

        # Display the results in the first column
        with col1:
            
            # Select a Algorithm
            choice2 = ["Logistic Regression (LR)", "Random Forest (RF)","Gradient Boosting (GB)", "Extreme Gradient Boosting (XGB)"]
            option2 = st.selectbox("Select a algorithm", choice2)
            
            # Logistic Regression
            if option2 == choice2[0]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 86.7% ")
                st.write("**Test Accuracy** : 86.9% " )
                st.write(" ")
                st.write(" ")
                
                t = {'class': [0, 1],
                        'precision': [0.78, 0.88],
                        'recall': [0.55, 0.96],
                        'f1 score': [0.65, 0.92]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # Random Forest   
            elif option2 == choice2[1]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 99.6% ")
                st.write("**Test Accuracy** : 88.4% " )
                st.write(" ")
                st.write(" ")
                
                t = {'class': [0, 1],
                        'precision': [0.78, 0.91],
                        'recall': [0.64, 0.95],
                        'f1 score': [0.71, 0.93]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
            
            # Gradient Boosting    
            elif option2 == choice2[2]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 87.9% ")
                st.write("**Test Accuracy** : 87.9% " )
                st.write(" ")
                st.write(" ")
                
                t = {'class': [0, 1],
                        'precision': [0.79, 0.89],
                        'recall': [0.59, 0.96],
                        'f1 score': [0.68, 0.93]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
            
            # Extreme Gradient Boosting    
            elif option2 == choice2[3]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 89.6% ")
                st.write("**Test Accuracy** : 89.1% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.82, 0.91],
                        'recall': [0.64, 0.96],
                        'f1 score': [0.72, 0.93]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
        
            
        # Display the image in the second column
        with col2:
            st.subheader(' ') 
            st.image("images/ROC meta.png", use_column_width=True)  
        
        
        # Use columns to display the results and image side by side
        col3, col4 = st.columns([1,1])

        with col3:
            st.image("images/Best.jpg", use_column_width=True) 
            # Add a centered caption to the image with an HTML link
            caption_text = 'Image by <a href="https://de.freepik.com/vektoren-kostenlos/illustration-des-mitarbeiterwertschaetzungskonzepts_40467235.htm#page=6&query=win&position=31&from_view=search&track=sph&uuid=9da1cffd-0a76-4a16-8aad-c94864c157b9" target="_blank">storyset on Freepik</a>'
            st.markdown(f'<div style="text-align:center">{caption_text}</div>', unsafe_allow_html=True)
        with col4:
            st.write(' ')
            st.write(' ')
            st.write('### Best model for metadata')
            st.write(' ')
            st.write('**Best model:** Extreme Gradient Boosting')
            my_text = """
            <p><strong>Best hyperparameter:</strong></p>
            <ul>
                <li>learning_rate = 0.1</li>
                <li>max_depth = 10</li>
                <li>subsample = 0.8</li>
                <li>colsample_bytree = 0.8</li>
                <li>n_estimators = 200</li>
                <li>gamma = 0.1</li>
            </ul>
            """
            st.markdown(my_text, unsafe_allow_html=True)
            st.write("**Train Accuracy** : 91.7% | **Test Accuracy** : 89.3%")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                'precision': [0.82, 0.91],
                'recall': [0.65, 0.96],
                'f1 score': [0.72, 0.93]}
            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)
            
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.title("Understanding Model Predictions with SHAP")

        st.write("SHAP (SHapley Additive exPlanations) helps interpret model predictions by highlighting the significance of individual features.")
        st.write(" ")
        st.write(" ") 
        # Use columns to display the results and image side by side
        col5, col6,col7 = st.columns([1,0.5,1.5])
        with col5:
            st.image("images/XGB SHAP.png", use_column_width=True)
        with col7:
            st.write(" ")
            st.write(" ")
            st.write(" ") 
            st.write(" ")
            # Explanation of the most influential feature
            st.write("**Key Feature: Num_of_words**")
            st.write("The number of words has the greatest impact on predictions. Longer comments often provide more context and nuances for sentiment analysis.")

            # Explanation of Capital_count
            st.write("**Capital Letters (Capital_count)**")
            st.write("The count of capital letters suggests strong emphasis or emotional intensity. Capitalization can indicate special attention or emphasis from the author.")

            # Explanation of Integer_count
            st.write("**Numbers in Text (Integer_count)**")
            st.write("The presence of numbers implies factual information or specific details. This is crucial in sentiment analysis, influencing the context of feedback.")


        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")  
        # Summary of SHAP analysis
        st.write("**SHAP Analysis Insights**")
        st.write("SHAP analysis confirms the importance of word count and provides deeper insights into feature meanings.")
        st.write("Understanding these features enhances our grasp of how they influence sentiment predictions.")
        
        
        
        
        
    elif option1 == choice1[1]:
        # Use columns to display the results and image side by side
        col1, col2 = st.columns([1,1])

        # Display the results in the first column
        with col1:
            
            # Select a Algorithm
            choice2 = ["Logistic Regression (LR_txt)", "Gradient Boosting (GB_txt)","Multinomial Naive Bayes (MNB)", "Extreme Gradient Boosting (XGB_txt)", "Balanced Random Forest (BRF)"]
            option2 = st.selectbox("Select a algorithm", choice2)
            
            # Select a Text Vectorizer
            choice3 = ["CountVectorizer","TFIDFVectorizer"]
            option3 = st.selectbox("Select a text vectorizer", choice3)
            
            # Select unigrams or bigrams
            choice4 = ["Unigrams","Bigrams"]
            option4 = st.selectbox("Select unigrams or bigrams", choice4)
            
        
            
            # CountVectorizer, 1gram
            # LR_txt
            if option4 == choice4[0] and option3 == choice3[0] and option2 == choice2[0]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 95.0% ")
                st.write("**Test Accuracy** : 93.3% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.80, 0.97],
                        'recall': [0.89, 0.94],
                        'f1 score': [0.84, 0.96]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # GB_txt
            elif option4 == choice4[0] and option3 == choice3[0] and option2 == choice2[1]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 88.6% ")
                st.write("**Test Accuracy** : 88.5% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.53, 0.99],
                        'recall': [0.91, 0.88],
                        'f1 score': [0.67, 0.93]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # MNB
            elif option4 == choice4[0] and option3 == choice3[0] and option2 == choice2[2]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 92.5% ")
                st.write("**Test Accuracy** : 92.0% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.81, 0.95],
                        'recall': [0.82, 0.95],
                        'f1 score': [0.82, 0.95]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # XGB_txt
            elif option4 == choice4[0] and option3 == choice3[0] and option2 == choice2[3]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 92.8% ")
                st.write("**Test Accuracy** : 92.1% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.73, 0.97],
                        'recall': [0.89, 0.93],
                        'f1 score': [0.80, 0.95]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
            
            # BRF
            elif option4 == choice4[0] and option3 == choice3[0] and option2 == choice2[4]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 96.3% ")
                st.write("**Test Accuracy** : 91.2% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.90, 0.92],
                        'recall': [0.75, 0.97],
                        'f1 score': [0.82, 0.94]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # CountVectorizer, 2gram
            # LR_txt_2
            if option4 == choice4[1] and option3 == choice3[0] and option2 == choice2[0]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 99.3% ")
                st.write("**Test Accuracy** : 93.8% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.82, 0.97],
                        'recall': [0.89, 0.95],
                        'f1 score': [0.85, 0.96]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # GB_txt_2
            elif option4 == choice4[1] and option3 == choice3[0] and option2 == choice2[1]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 87.9% ")
                st.write("**Test Accuracy** : 87.8% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.53, 0.98],
                        'recall': [0.91, 0.88],
                        'f1 score': [0.67, 0.93]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # MNB_2
            elif option4 == choice4[1] and option3 == choice3[0] and option2 == choice2[2]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 96.2% ")
                st.write("**Test Accuracy** : 93.0% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.82, 0.96],
                        'recall': [0.85, 0.95],
                        'f1 score': [0.84, 0.96]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # XGB_txt_2
            elif option4 == choice4[1] and option3 == choice3[0] and option2 == choice2[3]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 92.7% ")
                st.write("**Test Accuracy** : 92.1% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.73, 0.98],
                        'recall': [0.89, 0.93],
                        'f1 score': [0.80, 0.95]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
            
            # BRF_2
            elif option4 == choice4[1] and option3 == choice3[0] and option2 == choice2[4]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 96.6% ")
                st.write("**Test Accuracy** : 91.5% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.89, 0.92],
                        'recall': [0.76, 0.97],
                        'f1 score': [0.82, 0.94]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # TFIDFVectorizer, 1gram
            # LR_txt_TF
            if option4 == choice4[0] and option3 == choice3[1] and option2 == choice2[0]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 94.4% ")
                st.write("**Test Accuracy** : 93.7% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.83, 0.97],
                        'recall': [0.88, 0.95],
                        'f1 score': [0.85, 0.96]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # GB_txt_TF
            elif option4 == choice4[0] and option3 == choice3[1] and option2 == choice2[1]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 88.7% ")
                st.write("**Test Accuracy** : 88.6% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.54, 0.98],
                        'recall': [0.90, 0.88],
                        'f1 score': [0.67, 0.93]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # MNB_TF
            elif option4 == choice4[0] and option3 == choice3[1] and option2 == choice2[2]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 91.8% ")
                st.write("**Test Accuracy** : 91.1% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.65, 0.98],
                        'recall': [0.92, 0.91],
                        'f1 score': [0.76, 0.95]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # XGB_txt_TF
            elif option4 == choice4[0] and option3 == choice3[1] and option2 == choice2[3]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 93.1% ")
                st.write("**Test Accuracy** : 92.3% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.74, 0.97],
                        'recall': [0.88, 0.93],
                        'f1 score': [0.81, 0.95]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
            
            # BRF_TF
            elif option4 == choice4[0] and option3 == choice3[1] and option2 == choice2[4]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 95.2% ")
                st.write("**Test Accuracy** : 90.4% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.91, 0.90],
                        'recall': [0.72, 0.97],
                        'f1 score': [0.80, 0.94]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # TFIDFVectorizer, 2gram
            # LR_txt_TF_2
            if option4 == choice4[1] and option3 == choice3[1] and option2 == choice2[0]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 95.8% ")
                st.write("**Test Accuracy** : 93.9% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.83, 0.97],
                        'recall': [0.88, 0.95],
                        'f1 score': [0.86, 0.96]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # GB_txt_TF_2
            elif option4 == choice4[1] and option3 == choice3[1] and option2 == choice2[1]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 88.7% ")
                st.write("**Test Accuracy** : 88.6% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.53, 0.98],
                        'recall': [0.90, 0.88],
                        'f1 score': [0.67, 0.93]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # MNB_TF_2
            elif option4 == choice4[1] and option3 == choice3[1] and option2 == choice2[2]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 90.6% ")
                st.write("**Test Accuracy** : 87.8% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.48, 1.00],
                        'recall': [0.97, 0.87],
                        'f1 score': [0.64, 0.93]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
                
            # XGB_txt_TF_2
            elif option4 == choice4[1] and option3 == choice3[1] and option2 == choice2[3]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 93.2% ")
                st.write("**Test Accuracy** : 92.3% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.74, 0.97],
                        'recall': [0.89, 0.93],
                        'f1 score': [0.81, 0.95]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
            
            # BRF_TF_2
            elif option4 == choice4[1] and option3 == choice3[1] and option2 == choice2[4]:
                st.write(" ")
                st.write(" ")
                st.write("**Train Accuracy** : 95.5% ")
                st.write("**Test Accuracy** : 90.8% " )
                st.write(" ")
                st.write(" ")
                t = {'class': [0, 1],
                        'precision': [0.89, 0.91],
                        'recall': [0.74, 0.97],
                        'f1 score': [0.81, 0.94]}

                # Create a DataFrame
                T = pd.DataFrame(t)
                T=T.round(2).astype(str).reset_index(drop=True)
                # Display the table
                st.table(T)
  
        # Display the image in the second column
        with col2:
            if option3== choice3[0] and option4== choice4[0]:
                st.subheader(' ') 
                st.image("images/ROC txt.png", use_column_width=True) 
            
            elif option3== choice3[0] and option4== choice4[1]:
                st.subheader(' ') 
                st.image("images/ROC txt_2.png", use_column_width=True) 
                
            elif option3== choice3[1] and option4== choice4[0]:
                st.subheader(' ') 
                st.image("images/ROC txt_TF.png", use_column_width=True)
                
            elif option3== choice3[1] and option4== choice4[1]:
                st.subheader(' ') 
                st.image("images/ROC txt_TF_2.png", use_column_width=True)   
                
                 
        # Use columns to display the results and image side by side
        col3, col4 = st.columns([1,1])

        with col3:
            st.image("images/Best.jpg", use_column_width=True) 
            # Add a centered caption to the image with an HTML link
            caption_text = 'Image by <a href="https://de.freepik.com/vektoren-kostenlos/illustration-des-mitarbeiterwertschaetzungskonzepts_40467235.htm#page=6&query=win&position=31&from_view=search&track=sph&uuid=9da1cffd-0a76-4a16-8aad-c94864c157b9" target="_blank">storyset on Freepik</a>'
            st.markdown(f'<div style="text-align:center">{caption_text}</div>', unsafe_allow_html=True)
        with col4:
            st.write(' ')
            st.write(' ')
            st.write('### Best model for textdata')
            st.write(' ')
            st.write('**Best model:** Logistic Regression with TFIDFVecorizer and Bigrams')
            
            my_text = """
            <p><strong>Best hyperparameter:</strong></p>
            <ul>
                <li>C = 5</li>
                <li>class_weight = None</li>
                <li>max_iter = 4000</li>
                <li>solver = liblinear</li>
            </ul>
            """
            st.markdown(my_text, unsafe_allow_html=True)
            st.write("**Train Accuracy** : 98.6% | **Test Accuracy** : 94.2%")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                'precision': [0.85, 0.97],
                'recall': [0.88, 0.96],
                'f1 score': [0.86, 0.96]}
            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)   
            
            
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.title("Feature Importance for textdata")

        # Feature importance explanation
        st.write(
            "In a text classification model, the coefficients represent the weight of the words in the text. "
            "The larger the coefficient (positive or negative), the more the corresponding word or feature contributes to the prediction.")
        st.write(" ")
        st.write(" ") 
        # Use columns to display the results and image side by side
        col5, col6 = st.columns([1,1])
        with col5:
            st.image("images/LR Feature Importance.png", use_column_width=True)
        with col6:
            st.write(" ")
            st.write(" ")
            st.write(" ") 
            st.write(" ")
            # Explanation for positive and negative indicators
            st.header("Interpretation:")
            st.write(
                "The presence of words like 'excellent,' 'amazing,' 'best,' and 'great' significantly increases the probability "
                "of a positive evaluation. Users often use these words to express highly positive sentiments.")
            st.write(
                "The word 'easy' is also a strong positive indicator, suggesting that comments containing the word 'easy' "
                "are likely to be positively evaluated.")
            st.write(
                "'Poor,' 'worst,' and 'terrible' are strong negative indicators, suggesting that comments containing these words "
                "are likely to be negatively evaluated.")


    
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")  
        st.write(
            "The signs of the coefficients align with the intuitive understanding that certain words strongly influence "
            "the sentiment expressed in the text, either positively or negatively.")
  

elif page == pages[4] : 
    st.subheader('KMeans Clustering')
    st.write(" ")
    st.write(" ")
    st.write('''The goal of KMeans Clustering is not to classify the target variable, 
           but to bring similar data points into the same cluster and distinguish the data points in different clusters from each other.''')
    st.write('''The K-means algorithm was executed with 4 clusters. Subsequently, 
           a 2D visualization of the clustering results, based on the first two principal components, 
           was performed following PCA. The points in the scatterplot are color-coded according to their cluster membership. ''')
    st.write(' ')
    st.write(' ') 
    # Use columns to display the results and image side by side
    col1, col2 = st.columns([1,1])

    with col2:
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write(" ")
        st.write("The scatterplot illustrates significant overlap between clusters and suggests complicated comment patterns and subtleties. ")
        st.write(" ")
        st.write("Clusters 0, 1 and 2 can be clearly distinguished from each other, while in cluster 3 you only see isolated data points. ")
    
    with col1:
        st.image("images/KMeans Clustering.png", use_column_width=True)
        
    st.write(" ")
    st.write(" ")    
    st.subheader("Further research is required:")
    if  st.checkbox("Distribution of Data Points in Each Cluster"):
        # Use columns to display the results and image side by side
        col1, col2 = st.columns([1,1])

        with col1:
            st.image("images/Distribution of the Cluster.png", use_column_width=True)
        with col2:
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write('**Cluster 0:** majority of the data')
            st.write('**Cluster 1:** second largest data points')
            st.write('**Cluster 2:** specific subset of data points')
            st.write('**Cluster 3:** special group of comments that are significantly different from others')
        
    if  st.checkbox("Cluster characteristics"):
       # Use columns to display the results and image side by side
        col1, col2 = st.columns([1,3])

        with col1:
            st.image("images/c0.png", use_column_width=True)
            st.image("images/c1.png", use_column_width=True)
            st.image("images/c2.png", use_column_width=True)
            st.image("images/c3.png", use_column_width=True)
        
        with col2:
            st.write(' ')
            st.write('**Cluster 0:**')
            st.write('-	Primarily consists of positive reviews expressing high levels of customer satisfaction.')
            st.write('-	Commendations for product quality are frequent (e.g., index 3, 24, 25, 33).')
            st.write('-	Positive mentions of fast shipping are highlighted (e.g., index 17, 40).')
            st.write('-	Customers express positive experiences with various products and appreciate discounts (e.g., Index 48, 50).')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(' ')
            st.write(' ')
            st.write('**Cluster 1:**')
            st.write('-	Reflects mixed reviews that encompass both positive and negative experiences.')
            st.write('-	Negative comments are prevalent, particularly regarding defective products and communication problems (e.g., index 0, 11, 28, 29).')
            st.write('-	Positive reviews acknowledge fast shipping and positive interactions with customer service (e.g., Index 13, 14, 30).')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(" ")
            st.write(' ')
            st.write(' ')
            st.write('**Cluster 2:**')
            st.write('-	Consists of consistently positive reviews with a notable emphasis on low prices and fast shipping.')
            st.write('-	Customers praise competitive prices and product quality (e.g., index 4, 44, 45, 79).')
            st.write('-	Highlights satisfaction with the purchasing process and discounts offered (e.g., index 62, 90, 98).')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(' ')
            st.write(" ")
            st.write(" ")
            st.write(' ')
            st.write(' ')
            st.write(" ")
            st.write(" ")
            st.write(' ')
            st.write(' ')
            st.write('**Cluster 3:**')
            st.write('-	Comprises exclusively positive reviews, indicating long-term positive customer loyalty.')
            st.write('-	The word "always" appears in almost every comment, which suggests that customers, for example, buy a certain product again and again or shop with a certain company again and again (e.g. index 207,370,1578).')
            st.write('-	Customers express positive experiences with discounts and commend the quality of products (e.g., Index 74, 159, 599).')
            st.write('-	Emphasizes customer loyalty to the company over an extended period (e.g., index 430, 715).')
       
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.subheader('What makes each cluster unique:')
        st.write('	**Cluster 0:** Emphasizes the quality of products and customer satisfaction.')
        st.write('	**Cluster 1:** Contains mixed reviews with a focus on issues and customer service.')
        st.write('	**Cluster 2:** Shows positive reviews with a strong emphasis on price and fast shipping.')
        st.write('	**Cluster 3:** Contains consistently positive reviews that reflect long-term experiences.')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        st.write(' ')
        
    if  st.checkbox("Statistic information of each cluster"):
        col1, col2, col3 = st.columns([1,2,1])
        with col1:
             st.write(' ')
             
        with col2:
            st.image("images/statistic.png", use_column_width=True)
            
        with col3:
            st.write(' ') 
        
        # Use columns to display the results side by side
        col4, col5, col6, col7 = st.columns([1,1,1,1])

        with col4:
            st.markdown(
                """
                **Cluster 0:**
                - Highly positive emotional state (0.89) on average, indicating predominantly positive reviews.
                - Some variance in reviews suggests diverse customer experiences.
                - Most common rating: "great experience."
                - Uniqueness of reviews (162,161) indicates diverse experiences.
                """)
            
        with col5:
            st.markdown(
                """
                **Cluster 1:**
                - Average positive sentiment score: 0.49, indicating mixed or more neutral reviews.
                - High standard deviation (0.50) shows variance in emotional states.
                - Most common rating: "easy check."
                - Uniqueness of reviews (107,294) shows different impressions.
                """)
        
        with col6:
            st.markdown(
                """
                **Cluster 2:**
                - Extremely positive emotional state (0.98) on average, indicating predominantly positive reviews.
                - High uniformity in emotional states (standard deviation: 0.15).
                - Most common review: "great service."
                - High uniqueness of reviews (44,327) shows diverse positive experiences.
                """)
            
        with col7:
            st.markdown(
                  """
                **Cluster 3:**
                - Highest average positive affect (0.99), indicating extremely positive evaluations.
                - High uniformity in emotional states (standard deviation: 0.10).
                - Most common review: "always great experience."
                - Uniqueness of reviews (7,564) shows specific positive experiences.
                """)
        
        st.write(' ')
        st.write(' ')
        
        
    if  st.checkbox("Most common words of each cluster"):
        
         # Use columns to display the results side by side
        col1, col2,col3 = st.columns([0.5,4,0.5])
        with col1:
            st.write(' ')
        
          
        with col2:
            st.image("images/Most common words in each cluster.png", use_column_width=True) 
        with col3:
            st.write(' ')
           
             
        st.write(' ') 
        st.write(' ')     
        
        
    if  st.checkbox("T-test"):
        st.subheader('T-test between Cluster 0 and 2')
        st.image("images/t-test.png", width= 800) 
     
       
    st.write(' ') 
    st.write(' ')    
    # Title
    st.subheader("Utilizing Cluster Analysis Insights for Business Improvement")
    st.write(' ')   
    # Title, Example, and Benefit data
    data = [
        {"Title": "Customer Service Optimization",
        "Example": "We identify that Cluster 1 contains many mixed reviews due to communication issues.",
        "Benefit": "Targeted training for customer service in these specific areas could enhance customer satisfaction and minimize negative experiences."},
        
        {"Title": "Product Development and Customization",
        "Example": "Cluster 0 frequently highlights positive reviews about product quality.",
        "Benefit": "The company could use this insight to develop similar products or enhance existing ones to better meet customer expectations."},
    
        {"Title": "Marketing Strategies and Campaigns",
        "Example": "Cluster 2 emphasizes low prices and fast shipping.",
        "Benefit": "The company can highlight these positive aspects in marketing efforts, promoting special discounts or shipping advantages to attract customers."},
    
        {"Title": "Loyalty Programs and Rewards",
        "Example": "Cluster 3 reflects long-term positive experiences and frequently emphasizes the term 'always'.",
        "Benefit": "The company could introduce special loyalty programs or rewards for returning customers to further strengthen this positive connection."},
    
        {"Title": "Inventory Management and Product Placement",
        "Example": "Cluster 0 indicates high demand for specific products.",
        "Benefit": "The company could adjust its inventory management and prominently display these products to increase customer satisfaction and revenue."},
    
        {"Title": "Targeted Customer Engagement",
        "Example": "Cluster 2 prefers competitive prices.",
        "Benefit": "The company can develop targeted marketing campaigns for price-conscious customers and introduce special offers for specific products."},
    
        {"Title": "Competitive Analysis",
        "Example": "Analyzing differences between clusters can reveal strengths and weaknesses of competitors.",
        "Benefit": "The company can leverage these insights to strategically position itself and identify competitive advantages."}]

    # Convert data to DataFrame
    benefits = pd.DataFrame(data)

    # Display data as a table
    st.table(benefits)
    
    st.write(' ') 
    st.write(' ') 

    # Conclusion
    st.write("##### Conclusion:")
    st.write("By strategically applying these examples, a company can not only address individual customer needs but also optimize operational processes, enhance customer satisfaction, and ultimately improve market success.")
  
    
    
  
  
  
elif page == pages[5] : 
    
    st.write("### Neural network with text and metadata") 
    st.write(" ")
    st.write("##### Pre-processing")
    st.write(" ")
    
    # Pre-processing
    col1, col2, col3 = st.columns([1,1,1])
    with col1:
            st.markdown('**Metadata:**')
            st.markdown(
                """
                Pipeline for numeric data with
                - SimpleImputer()
                - StandardScaler()
                """)
              
    with col2:
            st.markdown('**Text:**')
            st.markdown(
                """
                Pipeline for text data with
                - TFIDFVectorizer()
                - TruncatedSVD()
                """)
    with col3:
            st.markdown('**Bring metadata and text together :**')
            st.markdown(' ')
            st.markdown(
                """
                - Using ColumnTransformer to apply the specified transformations to numeric and text data separately.
                
                """)
    st.write(" ")
    st.write(" ")
    st.write("##### Training process")
    st.write(" ")
    
    # Display the code
    st.code("""
    # Create a neuronal network
    model_nn = Sequential()

    # Add layers for the combined data
    model_nn.add(Dense(64, input_dim=X_train.shape[1], activation='relu'))
    model_nn.add(Dense(32, activation='relu'))
    model_nn.add(Dense(1, activation='sigmoid'))

    # Compile the model
    model_nn.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
    
    # Train the model
    history=model_nn.fit(X_train,
                        y_train,
                        epochs=5,
                        batch_size=32,
                        validation_data=(X_valid, y_valid))
    """, language='python')
    
    
    # Use columns to display the results and image side by side
    col4, col5 = st.columns([1,2])

    # Display the results in the first column
    with col4:
            
        # Select a PCA
        choice1 = ["100", "200"]
        option1 = st.selectbox("Select n_components for PCA", choice1)
            
        # Select epochs
        choice2 = ["5","10","15"]
        option2 = st.selectbox("Select epochs", choice2)
            
        # Select batch_size
        choice3 = ["32","64"]
        option3 = st.selectbox("Select batch_size", choice3)
            
        
        # 100,5,32    
        if option1 == choice1[0] and option2 == choice2[0] and option3 == choice3[0]:
            st.write(" ")   
            st.write("**Test Accuracy** : 93.9% " )
            st.write("**ROC-AUC** : 0.90 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                        'precision': [0.88, 0.95],
                        'recall': [0.83, 0.97],
                        'f1 score': [0.85, 0.96]}

            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)
                
        # 100,10,32    
        elif option1 == choice1[0] and option2 == choice2[1] and option3 == choice3[0]:
            st.write(" ")   
            st.write("**Test Accuracy** : 94.2% " )
            st.write("**ROC-AUC** : 0.91 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                    'precision': [0.88, 0.96],
                    'recall': [0.84, 0.97],
                    'f1 score': [0.86, 0.96]}
            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)
             
        # 100,15,32    
        elif option1 == choice1[0] and option2 == choice2[2] and option3 == choice3[0]:
            st.write(" ")    
            st.write("**Test Accuracy** : 94.2% " )
            st.write("**ROC-AUC** : 0.90 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                    'precision': [0.89, 0.95],
                    'recall': [0.83, 0.97],
                    'f1 score': [0.86, 0.96]}

            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)
                     
         # 100,5,64    
        elif option1 == choice1[0] and option2 == choice2[0] and option3 == choice3[1]:
            st.write(" ")
            st.write("**Test Accuracy** : 93.8% " )
            st.write("**ROC-AUC** : 0.90 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                    'precision': [0.87, 0.96],
                    'recall': [0.84, 0.96],
                    'f1 score': [0.85, 0.96]}

            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)
                    
        # 100,10,64    
        elif option1 == choice1[0] and option2 == choice2[1] and option3 == choice3[1]:
            st.write(" ")
            st.write("**Test Accuracy** : 94.0% " )
            st.write("**ROC-AUC** : 0.91 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                    'precision': [0.87, 0.96],
                    'recall': [0.85, 0.97],
                    'f1 score': [0.86, 0.96]}

            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)
                 
        # 100,15,64    
        elif option1 == choice1[0] and option2 == choice2[2] and option3 == choice3[1]:
            st.write(" ")
            st.write("**Test Accuracy** : 94.2% " )
            st.write("**ROC-AUC** : 0.90 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                    'precision': [0.89, 0.96],
                    'recall': [0.84, 0.97],
                    'f1 score': [0.86, 0.96]}

            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)
            
        # 200,5,32    
        elif option1 == choice1[1] and option2 == choice2[0] and option3 == choice3[0]:
            st.write(" ")
            st.write("**Test Accuracy** : 94.5% " )
            st.write("**ROC-AUC** : 0.91 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                    'precision': [0.90, 0.96],
                    'recall': [0.85, 0.97],
                    'f1 score': [0.87, 0.96]}

            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)
                    
        # 200,10,32    
        elif option1 == choice1[1] and option2 == choice2[1] and option3 == choice3[0]:
            st.write(" ")
            st.write("**Test Accuracy** : 94.5% " )
            st.write("**ROC-AUC** : 0.91 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                    'precision': [0.90, 0.96],
                    'recall': [0.84, 0.97],
                    'f1 score': [0.87, 0.97]}

            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T) 
            
        # 200,15,32    
        elif option1 == choice1[1] and option2 == choice2[2] and option3 == choice3[0]:
            st.write(" ")   
            st.write("**Test Accuracy** : 94.4% " )
            st.write("**ROC-AUC** : 0.91 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                    'precision': [0.89, 0.96],
                    'recall': [0.85, 0.97],
                    'f1 score': [0.87, 0.96]}

            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)
                     
         # 200,5,64    
        elif option1 == choice1[1] and option2 == choice2[0] and option3 == choice3[1]:
            st.write(" ")
            st.write("**Test Accuracy** : 94.3% " )
            st.write("**ROC-AUC** : 0.90 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                    'precision': [0.91, 0.95],
                    'recall': [0.82, 0.98],
                    'f1 score': [0.86, 0.96]}

            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)
                    
        # 200,10,64    
        elif option1 == choice1[1] and option2 == choice2[1] and option3 == choice3[1]:
            st.write(" ")    
            st.write("**Test Accuracy** : 94.5% " )
            st.write("**ROC-AUC** : 0.91 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                    'precision': [0.89, 0.96],
                    'recall': [0.85, 0.97],
                    'f1 score': [0.87, 0.96]}

            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)
                 
        # 200,15,64    
        elif option1 == choice1[1] and option2 == choice2[2] and option3 == choice3[1]:
            st.write(" ")
            st.write("**Test Accuracy** : 94.4% " )
            st.write("**ROC-AUC** : 0.90 ")
            st.write(" ")
            st.write(" ")
            t = {'class': [0, 1],
                    'precision': [0.92, 0.95],
                    'recall': [0.81, 0.98],
                    'f1 score': [0.86, 0.96]}

            # Create a DataFrame
            T = pd.DataFrame(t)
            T=T.round(2).astype(str).reset_index(drop=True)
            # Display the table
            st.table(T)  
            
    
    with col5:
        # 100,5,32    
        if option1 == choice1[0] and option2 == choice2[0] and option3 == choice3[0]:
            st.image("images/100-5-32.png", use_column_width=True)    
        # 100,10,32    
        elif option1 == choice1[0] and option2 == choice2[1] and option3 == choice3[0]:
            st.image("images/100-10-32.png", use_column_width=True)   
        # 100,15,32    
        elif option1 == choice1[0] and option2 == choice2[2] and option3 == choice3[0]:
            st.image("images/100-15-32.png", use_column_width=True)       
         # 100,5,64    
        elif option1 == choice1[0] and option2 == choice2[0] and option3 == choice3[1]:
            st.image("images/100-5-64.png", use_column_width=True)      
        # 100,10,64    
        elif option1 == choice1[0] and option2 == choice2[1] and option3 == choice3[1]:
            st.image("images/100-10-64.png", use_column_width=True)   
        # 100,15,64    
        elif option1 == choice1[0] and option2 == choice2[2] and option3 == choice3[1]:
            st.image("images/100-15-64.png", use_column_width=True)
                  
        # 200,5,32    
        elif option1 == choice1[1] and option2 == choice2[0] and option3 == choice3[0]:
            st.image("images/200-5-32.png", use_column_width=True)      
        # 200,10,32    
        elif option1 == choice1[1] and option2 == choice2[1] and option3 == choice3[0]:
            st.image("images/200-10-32.png", use_column_width=True)   
        # 200,15,32    
        elif option1 == choice1[1] and option2 == choice2[2] and option3 == choice3[0]:
            st.image("images/200-15-32.png", use_column_width=True)       
         # 200,5,64    
        elif option1 == choice1[1] and option2 == choice2[0] and option3 == choice3[1]:
            st.image("images/200-5-64.png", use_column_width=True)      
        # 200,10,64    
        elif option1 == choice1[1] and option2 == choice2[1] and option3 == choice3[1]:
            st.image("images/200-10-64.png", use_column_width=True)   
        # 200,15,64    
        elif option1 == choice1[1] and option2 == choice2[2] and option3 == choice3[1]:
            st.image("images/200-15-64.png", use_column_width=True)   
          
    st.write(" ")
    st.write(" ")        
    st.write("##### Feature importance")       
    st.write(" ")    
     # Use columns to display the results and image side by side
    col6, col7 = st.columns([1,1])

    # Display the results in the first column
    with col6:
        st.image("images/NN num feature.png", use_column_width=True) 
        
    
    with col7: 
        st.image("images/NN text feature.png", use_column_width=True)          
            
    st.write(' ') 
    st.write(' ')    
    # Title
    st.write("##### Benefits for Companies")
    st.write(' ')   
    # Title, Example, and Benefit data
    data = [
        {"Title": "Swift Response to Negative Reviews",
        "Insight": "Comments with words like 'worst' and 'terrible' have a strong negative impact",
        "Benefit": "Companies can identify these comments promptly and respond quickly to rectify negative experiences, thereby protecting their brand image."},
        
        {"Title": "Product Improvements",
        "Insight": "Numerical features such as the number of words and the presence of capital letters influence predictions.",
        "Benefit": "Companies can leverage these features to infer the intensity of customer opinions and make targeted product improvements"},
    
        {"Title": "Targeted Marketing",
        "Insight": "Key words like great' have a positive influence.",
        "Benefit": "Positive words can be highlighted in marketing materials to strengthen the positive image and attract potential customers."},
    
        {"Title": "Improved Response to Customer Inquiries",
        "Insight": "Words like 'told', 'email', 'call' may indicate specific customer interactions.",
        "Benefit": "The company can better respond to specific inquiries, optimizing customer service."}]
    
    
    # Convert data to DataFrame
    benefits = pd.DataFrame(data)

    # Display data as a table
    st.table(benefits)
    

  
elif page == pages[6] :
    # Set page title
    st.header("Conclusion")
        
    # Introduction
    st.write("This report covered key themes, from data exploration to text mining and machine learning models. Here are the insights:")
    
    # Use columns to display the results and image side by side
    col1, col2 = st.columns([1,1])
    with col2:
        # Display the image
        image_path = "images/Goals.jpg"
        st.image(image_path, use_column_width=True)

        # Add a centered caption to the image using HTML and CSS
        caption_text = 'Image by <a href="https://www.freepik.com/free-vector/businessman-with-trophy-running-up-stairs-and-growth-chart-business-success-leadership-business-assets-and-planning-concept-on-white-background_11667202.htm" target="_blank">vectorjuice on Freepik</a>'
        st.markdown(f'<div style="text-align:center">{caption_text}</div>', unsafe_allow_html=True)
    
    with col1:   
        
        st.write(' ') 
        st.write(' ')
        st.write(' ') 
        st.write(' ')
        st.write(' ') 
        # Achieved Project Goals
        st.write("##### Achieved Project Goals:")
        st.markdown("- **Effective Data Preparation:** Thorough exploration and preparation laid the foundation for reliable analyses.")
        st.markdown("- **Accurate Sentiment Classification:** Various models demonstrated high accuracy and performance on the test dataset.")
        st.markdown("- **High Accuracy:** The models achieve a remarkable accuracy of up to 94% and equally high values for precision, recall and F1-score, even for the class with the bad reviews. ")
        st.markdown("- **Effective Visualization:** Feature importance and SHAP analysis provided comprehensive insights.")
        st.write(' ') 
        st.write(' ') 
        st.write(' ') 
        st.write(' ') 
    
    # Future Improvement Suggestions
    st.write("##### Future Improvements:")
    st.markdown("Despite success, there is room for future enhancements:")
    st.markdown("- **Data Quality:** Continuous monitoring and improvement are crucial.")
    st.markdown("- **Expansion of Data Sources:** Including additional sources could improve model accuracy.")
    st.markdown("- **Experimentation with Models:** Trying different models and optimization techniques could lead to more advanced results.")
    st.markdown("- **Ensemble Learning:** Combining predictions of multiple models may enhance overall accuracy.")
    st.markdown("- **Advanced Neural Network Architectures:** Experimenting with complex architectures could capture intricate patterns.")
    st.markdown("- **Real-time Sentiment Analysis:** Developing a system for immediate insights into customer sentiment.")
    st.markdown("- **Feedback Loop Integration:** Establishing a mechanism to use model insights for process and service improvements.")
    st.write(' ') 
    st.write(' ') 
    # Overall Conclusion
    st.markdown("This project achieved its goals, providing a clear path for leveraging machine learning and text analysis in review classification. The gained insights should help businesses strengthen customer relationships and optimize services.")
  

elif page == pages[7] : 
    st.write("### Write your own comment")
  
    
  

    # Load the trained model
    best_lr_txt_TF_2 = load('joblib/best_lr_txt_TF_2.pkl') 
    vectorizer_tf_2 = load('joblib/vectorizer_tf_2.pkl')


    # Generates stopwords
    stop_words = set(stopwords.words('english'))
    stop_words.update([",", ".", "?", ":", "!", "’", "/", ")", "'s", "(", "hair", "n't", "ghd"])

    # Function to filter stopwords
    def stop_words_filtering(text):
        text = str(text).lower()
        tokens = text.split()
        tokens_filtered = [word for word in tokens if word not in stop_words]
        return " ".join(tokens_filtered)

    # Function to clean the text
    def review_cleaning(text):
        text = str(text).lower()
        text = re.sub('\[.*?\]', '', text)
        text = re.sub('https?://\S+|www\.\S+', '', text)
        text = re.sub('<.*?>+', '', text)
        text = re.sub('\n', '', text)
        text = re.sub('\w*\d\w*', '', text)
        text = re.sub('[^\w\s]', ' ', text)
        return text

    # Function to perform lemmatization operations
    def lemmatization(text):
        wordnet_lemmatizer = WordNetLemmatizer()
        words = text.split()
        lemmatized = [wordnet_lemmatizer.lemmatize(word) for word in words]
        return ' '.join(lemmatized)
    
    

    # Function for custom preprocessing
    def custom_preprocessing(text, vectorizer):
        processed_text = stop_words_filtering(text)
        processed_text = review_cleaning(processed_text)
        processed_text = lemmatization(processed_text)

        # Transform the text using the TfidfVectorizer
        processed_vector = vectorizer.transform([processed_text])
        

        return processed_vector

    # Streamlit App
    def main():
        # Streamlit page title
        st.title("Sentiment Prediction")

        # Text input field in Streamlit
        comment = st.text_area("Enter your comment:", "")

        # Button to trigger prediction
        if st.button("Predict Sentiment"):
            # Perform specific preprocessing
            preprocessed_vector = custom_preprocessing(comment, vectorizer_tf_2)
    
            # Perform model prediction probabilities
            probabilities = best_lr_txt_TF_2.predict_proba(preprocessed_vector)[0]
    
            # Extract the probability for the positive class (class 1)
            positive_probability = probabilities[1]
            
            st.image("images/prediction.png", use_column_width=True)
            # Create a horizontal color bar
            fig, ax = plt.subplots(figsize=(8, 1))
            cmap = plt.get_cmap('RdYlGn')
            norm = plt.Normalize(0, 1)
            color_bar = plt.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=ax, orientation='horizontal')
            color_bar.set_label('Predicted Probability')

            # Add a black line at the predicted probability
            ax.axvline(positive_probability, color='black', linestyle='--', linewidth=2)

            # Add an arrow to indicate the predicted probability
            plt.annotate(f'{positive_probability:.2f}', xy=(positive_probability, 0.5), xytext=(0.5, 0.5), 
                         ha='center', va='center', fontsize=12, color='black')

            # Remove axis labels and ticks
            plt.xticks([])
            plt.yticks([])
            plt.xlabel('')
            plt.ylabel('')

            # Display the plot in Streamlit
            st.pyplot(fig)

    if __name__ == "__main__":
        main()
  
