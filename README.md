# Disaster Tweets NLP
 
### Brief description of the problem and data

Briefly describe the challenge problem and NLP. Describe the size, dimension, structure, etc., of the data. 

The challenge problem involves natural language processing (NLP) techniques to classify tweets into two categories: disaster tweets and non-disaster tweets. The dataset consists of two main components: a training set and a test set. The training set contains 7,613 rows and 5 columns, while the test set contains 3,263 rows and 4 columns. Each row represents a tweet, and the columns include 'id', 'keyword', 'location', 'text', and 'target' (only available in the training set).

The 'text' column contains the actual tweet text, while the 'target' column in the training set indicates whether the tweet is related to a disaster (target=1) or not (target=0). The 'keyword' and 'location' columns provide additional contextual information about the tweet, such as keywords associated with the tweet or the location from which it was posted.

The data has missing values in the 'keyword' and 'location' columns, with non-null counts indicating the presence of missing values. The 'keyword' column has 7552 non-null values in the training set and 3237 non-null values in the test set, while the 'location' column has 5080 non-null values in the training set and 2158 non-null values in the test set.

The structure of the data is tabular, with rows representing individual tweets and columns representing different attributes of the tweets. The 'text' column, which contains the tweet text, is of primary interest for NLP tasks such as sentiment analysis or classification. Additionally, the 'keyword' and 'location' columns may provide useful contextual information for improving model performance. Overall, the task involves leveraging NLP techniques to analyze and classify tweet data based on their content into disaster and non-disaster categories.

### Exploratory Data Analysis (EDA) — Inspect, Visualize and Clean the Data

Show a few visualizations like histograms. Describe any data cleaning procedures. Based on your EDA, what is your plan of analysis? 

We begins with an examination of the class distribution in the target variable, revealing a slight imbalance between disaster and non-disaster tweets, with 43% classified as disaster tweets and 57% as non-disaster tweets. Further investigation into tweet locations highlights urban hubs like New York and California as major contributors to tweet activity, both in disaster and non-disaster contexts. Additionally, global centers like the United States and the United Kingdom exhibit significant tweet volumes, indicating the global reach of Twitter in capturing real-time events.

Next, the analysis delves into the textual characteristics of disaster and non-disaster tweets, revealing nuanced differences in word length density, unique word count, and stop word distribution. Disaster tweets tend to be more densely packed with information, with higher word densities and longer mean word lengths compared to non-disaster tweets. Moreover, disaster tweets exhibit more intense bursts of punctuation and a higher frequency of hashtags and mentions, indicating differences in communication styles between the two categories.

Further exploration through n-gram analysis uncovers distinct patterns in word usage between disaster and non-disaster contexts. Disaster-related terms like 'wildfire' and 'suicide bomber' dominate disaster tweets, reflecting the urgency and gravity of such situations. Conversely, non-disaster tweets feature more commonplace language, with phrases like 'liked a youtube video' and 'reddit will now quarantine' prevailing in the discourse.

To prepare the text data for analysis, several cleaning procedures are applied, including lowercasing, punctuation removal, URL removal, HTML tag removal, emoji removal, stop word removal, and stemming. These preprocessing steps ensure standardized text data, facilitating downstream analysis tasks such as sentiment analysis or text classification.

Overall, the EDA provides valuable insights into the characteristics of disaster and non-disaster tweets, laying the groundwork for subsequent analysis tasks like sentiment analysis, topic modeling, or predictive modeling to classify tweets into their respective categories.

### Model Architecture

Describe your model architecture and reasoning for why you believe that specific architecture would be suitable for this problem. 

Since we did not learn NLP-specific techniques such as word embeddings in the lectures, we recommend looking at Kaggle tutorials, discussion boards, and code examples posted for this challenge.  You can use any resources needed, but make sure you “demonstrate” you understood by including explanations in your own words. Also importantly, please have a reference list at the end of the report.  

There are many methods to process texts to matrix form (word embedding), including TF-IDF, GloVe, Word2Vec, etc. Pick a strategy and process the raw texts to word embedding. Briefly explain the method(s) and how they work in your own words.

Build and train your sequential neural network model (You may use any RNN family neural network, including advanced architectures LSTM, GRU, bidirectional RNN, etc.). 

The chosen approach for text vectorization is based on the CountVectorizer method, which converts the raw text data into a matrix of token counts. This method transforms the text data into a numerical representation by counting the occurrences of each word in the vocabulary within each document (tweet). The resulting matrix, known as a document-term matrix (DTM), is then used as input for machine learning models.

To handle the textual data, a sequential neural network model architecture is employed, specifically utilizing LSTM (Long Short-Term Memory) units, which are a type of recurrent neural network (RNN). LSTMs are well-suited for NLP tasks because they can capture long-range dependencies and maintain information over extended sequences, making them effective for processing text data.

The preprocessing steps involve tokenizing the text, padding sequences to ensure uniform length, and embedding the sequences using pre-trained GloVe (Global Vectors for Word Representation) word embeddings. GloVe embeddings provide vector representations for words based on their co-occurrence statistics in a large corpus of text, capturing semantic relationships between words. These pre-trained embeddings are then used to initialize an embedding layer in the neural network model, enabling the model to leverage semantic information from the embeddings during training.

Hyperparameter tuning is performed to optimize the model's performance, including parameters such as recurrent dropout, dropout rate, LSTM units, and optimizer choice. The best-performing model architecture is selected based on its accuracy on a validation set. In addition to LSTM models, a GRU (Gated Recurrent Unit) architecture is also explored for comparison.

Evaluation metrics such as loss, accuracy, and learning curves (training/validation loss and accuracy over epochs) are analyzed to assess the model's performance and identify potential issues such as overfitting. Techniques like learning rate reduction and model checkpointing are employed to mitigate overfitting and improve generalization ability.

Overall, the selected model architecture (LSTM with better accuracy score) and preprocessing approach aim to effectively capture the semantic meaning of tweets and classify them into disaster and non-disaster categories with high accuracy. 

### Results and Analysis

Run hyperparameter tuning, try different architectures for comparison, apply techniques to improve training or performance, and discuss what helped.

Includes results with tables and figures. There is an analysis of why or why not something worked well, troubleshooting, and a hyperparameter optimization procedure summary.

In the pursuit of optimizing model performance for tweet classification into disaster and non-disaster categories, several strategies were employed, including hyperparameter tuning and exploring different model architectures.

Initially, a bidirectional LSTM model with GloVe embeddings was implemented. Hyperparameter tuning was conducted to find the optimal combination of parameters such as recurrent dropout, dropout rate, LSTM units, and optimizer. Through this process, the best hyperparameters were identified as a recurrent dropout of 0.2, dropout rate of 0.4, 128 LSTM units, and RMSprop optimizer, achieving an accuracy of 74.53%. Additionally, learning rate reduction and model checkpointing techniques were utilized to improve training efficiency and save the best model weights based on validation loss.

Visualization of training and validation metrics revealed potential overfitting issues as the validation loss began diverging from the training loss after the fifth epoch. Despite this, the model demonstrated an overall improvement in training accuracy, suggesting effective learning. However, the fluctuating validation accuracy and increasing validation loss indicated a loss of generalization ability, highlighting the need for further optimization.

To explore alternative architectures, a GRU (Gated Recurrent Unit) model was implemented, focusing on sentiment analysis. Hyperparameter tuning was conducted for dropout rates, batch sizes, and optimizers, such as SGD and Adagrad. Despite efforts to optimize, the GRU model achieved a lower accuracy of 56.36%, indicating that the LSTM architecture performed better for this specific task.

Visualizing the training and validation metrics for the GRU model showed consistent decreases in both training and validation loss, indicating effective learning. However, the model's performance plateaued, with minimal improvement in validation accuracy beyond a certain point, suggesting potential overfitting to the training data.

In summary, while hyperparameter tuning and exploring different architectures are essential for optimizing model performance, careful monitoring of training and validation metrics is crucial for identifying potential issues such as overfitting. Further optimization and experimentation may be necessary to improve the model's generalization ability and achieve higher accuracy.

### Conclusion

Discuss and interpret results as well as learnings and takeaways. What did and did not help improve the performance of your models? What improvements could you try in the future?

In conclusion, the experimentation with various model architectures and preprocessing techniques yielded valuable insights into text classification for disaster detection. The LSTM model with GloVe embeddings achieved the highest accuracy, indicating its effectiveness in capturing the semantic meaning of tweets and distinguishing between disaster-related and non-disaster-related content. The hyperparameter tuning process helped in optimizing model performance, with the selected hyperparameters significantly influencing the model's accuracy.

One notable finding was the impact of pre-trained GloVe embeddings on model performance. Leveraging semantic information encoded in the embeddings improved the model's ability to generalize to unseen data, resulting in better classification accuracy. Additionally, the use of LSTM units allowed the model to capture long-range dependencies in the text data, contributing to its overall effectiveness.

However, the analysis also revealed challenges such as potential overfitting, as indicated by diverging training and validation loss curves. While techniques like learning rate reduction and model checkpointing were employed to mitigate overfitting, further exploration of regularization techniques such as dropout and early stopping could be beneficial in improving model generalization.

Furthermore, the comparison with a GRU model highlighted the importance of architecture selection in achieving optimal performance. While the LSTM model outperformed the GRU model in terms of accuracy, the GRU model still provided valuable insights into alternative architectures for text classification tasks.

Moving forward, additional avenues for improvement could include exploring ensemble methods to combine multiple models for enhanced performance, experimenting with different pre-processing techniques such as stemming and lemmatization, and incorporating domain-specific features or external datasets to further enhance model robustness. Additionally, ongoing monitoring and fine-tuning of model hyperparameters based on real-world performance feedback could help ensure continued effectiveness in disaster detection applications. Overall, this experimentation process served as a valuable learning experience, highlighting the importance of iterative refinement and exploration in the pursuit of accurate and reliable text classification models.
