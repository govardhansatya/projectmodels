import numpy as np
import pandas as pd 
usrhist = pd.read_csv("D:\\acm\\models\\modelsong\\User Listening History.csv\\User Listening History.csv")
df  = pd.read_csv("D:\\acm\\models\\modelsong\\Music Info.csv\\Music Info.csv")
usrsonglist = usrhist.groupby('user_id', observed=True)[['track_id', 'playcount']].apply(lambda x: list(zip(x['track_id'], x['playcount']))).to_dict()
usrsonglist = {user: songs for user, songs in usrsonglist.items() if len(songs) >= 50}
usrhist = usrhist[usrhist['user_id'].isin(usrsonglist.keys())] 
usrhist.shape
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from sklearn.model_selection import train_test_split
from scipy.sparse import csr_matrix,coo_matrix
from annoy import AnnoyIndex
ftrcols = ['danceability', 'energy', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']
#normalizing the features
from sklearn.preprocessing import StandardScaler


scaler = StandardScaler()
normalisednumftrs = scaler.fit_transform(df[ftrcols])
felen = normalisednumftrs.shape[1]
annoy_index = AnnoyIndex(felen, 'angular')

for idx, vector in enumerate(normalisednumftrs):
    annoy_index.add_item(idx, vector)

annoy_index.build(50)

annoy_index.get_nns_by_item(0, 7)[1:]

train, test = train_test_split(usrhist, test_size=0.2, random_state=42)
train['user_id'] = train['user_id'].astype('category')
train['track_id'] = train['track_id'].astype('category')
usridmapping = dict(enumerate(train['user_id'].cat.categories))
trckidmapping = dict(enumerate(train['track_id'].cat.categories))
usridrevrsemapping = {v: k for k, v in usridmapping.items()}
trckidrevrsemapping = {v: k for k, v in trckidmapping.items()}

df['track_id'] = df['track_id'].astype('category')

cbtrckidmapping = dict(enumerate(df['track_id'].cat.categories))
cbtrckidrevrsemapping = {v: k for k, v in cbtrckidmapping.items()}
usersparseitem = coo_matrix((train['playcount'], (train['user_id'].cat.codes, train['track_id'].cat.codes)))
svd = TruncatedSVD(n_components=10, random_state=42)
usrfctrs = svd.fit_transform(usersparseitem)
itemfctrs = svd.components_.T
usersparseitem.shape
def reccsongshybrid(user_id, usritemmatrix, usrfctrs, itemfctrs, df, annoy_index, n_recommendations=5):
    usercode = usridrevrsemapping.get(user_id)
    if usercode is None:
        print(f"User ID {user_id} not found in the user-item matrix.")
        return []
    
    # Collaborative Filtering Recommendations
    cfpred = np.dot(usrfctrs[usercode, :], itemfctrs.T)
    cfindices = np.argsort(cfpred)[::-1]
    cfrecctrcks = [trckidmapping[i] for i in cfindices[:n_recommendations]]
    cbrecctrcks = []
    for track in cfrecctrcks:
        track_code = cbtrckidrevrsemapping.get(track)
        if track_code is not None:
            similar_tracks = annoy_index.get_nns_by_item(track_code, 4)[1:]  
            for i in similar_tracks:
                try:
                    cbrecctrcks.append(cbtrckidmapping[i])
                except KeyError:
                    print(f"KeyError: Index {i} not found in cbtrckidmapping")    
    hybrid_recommended_tracks = list(set(cfrecctrcks + cbrecctrcks))    
    return hybrid_recommended_tracks
def evalmodelhybrduser(user, user_test_data, user_train_data, usritemmatrix, usrfctrs, itemfctrs, df, annoy_index, n_recommendations=5):
    precision = 0.0
    recall = 0.0
    if user in user_train_data:
            truetrcks = user_test_data[user]
            recctrcks = reccsongshybrid(user, usritemmatrix, usrfctrs, itemfctrs, df, annoy_index, n_recommendations)
            tp = len(set(recctrcks) & set(truetrcks))
            precision = tp / len(recctrcks) if recctrcks else 0
            recall = tp / len(truetrcks) if truetrcks else 0
    return precision, recall  
def evalmodelhybrd(user_test_data, user_train_data, usritemmatrix, usrfctrs, itemfctrs, df, annoy_index, n_recommendations=5):
    precisions = []
    recalls = []

    for user, truetrcks in user_test_data.items():
        if user in user_train_data:
            recctrcks = reccsongshybrid(user, usritemmatrix, usrfctrs, itemfctrs, df, annoy_index, n_recommendations)
            tp = len(set(recctrcks) & set(truetrcks))
            precision = tp / len(recctrcks) if recctrcks else 0
            recall = tp / len(truetrcks) if truetrcks else 0
            
            precisions.append(precision)
            recalls.append(recall)
    
    avg_precision = np.mean(precisions)
    avg_recall = np.mean(recalls)
user_train_data = train.groupby('user_id', observed=True)['track_id'].apply(list).to_dict()
user_test_data = test.groupby('user_id', observed=True)['track_id'].apply(list).to_dict()
user_id = input("Enter user ID: ")
if user_id in user_train_data:
    print(reccsongshybrid(user_id, usersparseitem, usrfctrs, itemfctrs, df, annoy_index, n_recommendations=1))
else:
	print(f"User ID {user_id} not found in data.")