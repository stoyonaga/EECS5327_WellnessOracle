# WellnessOracle: A Machine Learning Approach to Suicide Ideation Classification

*WellnessOracle* is a text-based binary classifier that aims to predict a social media user's mental well-being based on their previous posting and/or comment history. It aims to determine signs of underlying depression so that the machine learning model can quickly determine if a user is mentally at-risk to themselves or others. Suicide, especially for the younger generation, has become an increasingly pervasive problem [8]. It has become the second-most leading cause of fatality [5] among young adults worldwide.

## Technical Details
1. Machine Learning
   - **sklearn, seaborn, GridSearchCV** were used to develop and fine-tune the model for text-based classification tasks.
2. Data Visualization
   - **matplotlib, seaborn, wordcloud** were used to visualize model performance on the training and testing set and to identify patterns between keywords.
3. Web Scraping
   - **PlayWright, BeautifulSoup4, Requests** were used to webscrape content form Reddit, Twitter, and Quora to further evaluate the model;s performance on real-time data.
4. Graphical User Interface (GUI)
   - **Streamlit** was used to support the end-user with an intuitive user interface that interacts with our pre-trained models. Data from various websites can be scraped and automatically classified as suicide or non-suicide.
  
## Dependencies
- The Suicide and Depression Detection Dataset was too large to upload (even after zipping!). You can download the file [here](https://www.kaggle.com/datasets/nikhileswarkomati/suicide-watch).
- The Streamlit UI will require the models being saved as a .pkl file. This can be done by modifying the Notebook (slightly), or you can reach out to one of the authors for a link to the Google Drive. Unfortunately, the files were too large to upload on GitHub.
