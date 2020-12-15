from flask import Flask, render_template, session, redirect, url_for, session
from flask_wtf import FlaskForm
from wtforms import TextField,SubmitField, StringField
from wtforms.validators import DataRequired
import pickle
from scipy import sparse
import pandas as pd
from fuzzywuzzy import fuzz
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors
import string
from flask_bootstrap import Bootstrap



def fuzzy_matching(fav_book, mapper, verbose=True):
    match_tuple = []

    for title, idx in mapper.items():
        ratio = fuzz.ratio(title.lower(), fav_book.lower())
        if ratio >= 60:
            match_tuple.append((title, idx, ratio))
            
    match_tuple = sorted(match_tuple, key=lambda x: x[2])[::-1]
    if not match_tuple:
        return
    return match_tuple[0][1]

def make_recommendation(model_knn, mapper, sample_json, data):
    title = sample_json['book_name']
    idx = fuzzy_matching(title, mapper)
    distances, indices = model_knn.kneighbors(data[idx], n_neighbors=11)
    
    raw_recommends =  sorted(list(zip(indices.squeeze().tolist(), distances.squeeze().tolist())), key=lambda x: x[1])[:0:-1]

    reverse_mapper = {v: k for k, v in mapper.items()}

    rec=[]
    for i, (idx, dist) in enumerate(raw_recommends):
        if idx not in reverse_mapper.keys():
            continue
        rec.append(reverse_mapper[idx])
    return rec

app = Flask(__name__)

# Configure a secret SECRET_KEY
app.config["SECRET_KEY"] = "someRandomKey"

Bootstrap(app)

# Loading the model and scaler
knnmodel = pickle.load(open('knn_model.pkl','rb'))
mapper = pd.read_pickle('indices.pkl')
data = sparse.load_npz("data.npz")

# Now create a WTForm Class
class BookForm(FlaskForm):
    book_name = StringField("Enter the name of a book you loved:", validators=[DataRequired()])
    submit = SubmitField("Recommend Me Books!")
 
@app.route('/', methods=['GET', 'POST'])
def submit():
  # Create instance of the form.
    form = BookForm()
  # If the form is valid on submission

    if form.validate_on_submit():
        #print("here")

  # Grab the data from the input on the form.
        session['book_name'] = form.book_name.data
        return redirect(url_for('prediction'))

    print("not valid")
    return render_template('home2.html', form=form)

@app.route('/prediction')
def prediction():
 #Defining content dictionary
    content = {}
    content['book_name'] = str(session['book_name'])
 
    results = make_recommendation(knnmodel, mapper, content, data)
    return render_template('prediction2.html',results=results)

if __name__ == '__main__':
 app.run(debug=True)

