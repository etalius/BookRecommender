import string
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from scipy import sparse
from sklearn.neighbors import NearestNeighbors
from fuzzywuzzy import fuzz
import pickle



books= pd.read_csv('books.csv',error_bad_lines = False)
ratings = pd.read_csv('ratings.csv')
ratings = ratings.sort_values("user_id")
tags = pd.read_csv('book_tags.csv')
i_to_tag = pd.read_csv('tags.csv')

ratings.drop_duplicates(subset=["user_id","book_id"], keep = False, inplace = True) 
books.drop_duplicates(subset = 'original_title', keep=False, inplace=True)
i_to_tag.drop_duplicates(subset='tag_id', keep=False, inplace=True)
tags.drop_duplicates(subset=['goodreads_book_id', 'tag_id'], keep=False, inplace=True)


books_col = books[['book_id', 'original_title']]
books_col.dropna()
num_users = len(ratings.user_id.unique())
num_items = len(ratings.book_id.unique())
ratings = ratings.dropna()

temp_count = pd.DataFrame(ratings.groupby('rating').size(), columns=['count'])

total = num_users * num_items
zero_count = total - ratings.shape[0]

rating_counts = temp_count.append(pd.DataFrame({'count': zero_count}, index=[0.0]), verify_integrity=True,).sort_index()
df_books_cnt = pd.DataFrame(ratings.groupby('book_id').size(), columns=['count'])
book_thres = 50
pop_books = list(set(df_books_cnt.query('count >= @book_thres').index))
df_ratings_drop = ratings[ratings.book_id.isin(pop_books)]
df_users_cnt = pd.DataFrame(df_ratings_drop.groupby('user_id').size(), columns=['count'])
user_thres = 30
active_users = list(set(df_users_cnt.query('count >= @user_thres').index))
df_ratings_drop_users = df_ratings_drop[df_ratings_drop.user_id.isin(active_users)]
book_user_mat = df_ratings_drop_users.pivot(index='book_id', columns='user_id', values='rating').fillna(0)
book_user_mat_sparse = csr_matrix(book_user_mat.values)
model_knn = NearestNeighbors(metric='cosine', algorithm='brute', n_neighbors=20, n_jobs=-1)

model_knn.fit(book_user_mat_sparse)

sparse.save_npz("data.npz", book_user_mat_sparse)

indices = pd.Series(books_col.index, index=books_col['original_title'])
indices.to_pickle('indices.pkl')

with open("knn_model.pkl", "wb") as file_handler:
    pickle.dump(model_knn, file_handler)

print("done")

 