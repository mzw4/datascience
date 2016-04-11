
# coding: utf-8

# In[2]:

import sys, operator, time, os
from collections import defaultdict
import numpy as np

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

sc = SparkContext(appName="PythonPi")

# %matplotlib inline
# print os.environ["PYSPARK_SUBMIT_ARGS"]
# print sc._conf.get('spark.driver.memory')
# print sc._conf.get('spark.executor.memory')
# print sc._conf.getAll()

base_dir = 'ml-10M100K/'
base_dir = 's3n://mzw4cs5304hw2/'
ratings_dir = base_dir + 'ratings.dat'
movies_dir = base_dir + 'movies.dat'
tags_dir = base_dir + 'tags.dat'

debug = False


# In[3]:

# load data as rdd
ratings_rdd = sc.textFile(ratings_dir)    .map(lambda r: [float(e) if i == 2 else e for i, e in enumerate(r.split('::'))])
# movies_rdd = sc.textFile(movies_dir).map(lambda r: r.split('::'))
# tags_rdd = sc.textFile(tags_dir).map(lambda r: r.split('::'))


# In[4]:

# for testing
if debug:
    ratings_rdd = ratings_rdd.sample(False, 0.001, int(time.time()))


# In[ ]:

print 'Sorting by timestamp'
ratings_rdd = ratings_rdd.sortBy(lambda x: x[3])


# In[33]:

print 'Split dataset'
# find timestamp at 60% and 80% to split rdd
size=ratings_rdd.count()
divider_60 = ratings_rdd.map(lambda x: x[3]).take(int(0.6*size))[-1]
divider_80 = ratings_rdd.map(lambda x: x[3]).take(int(0.8*size))[-1]
train_rdd=ratings_rdd.filter(lambda x: x[3]<divider_60)
validation_rdd=ratings_rdd.filter(lambda x: x[3]>=divider_60 and x[3]<divider_80)
test_rdd=ratings_rdd.filter(lambda x: x[3]>=divider_80)


# In[34]:

print train_rdd.count(), validation_rdd.count(), test_rdd.count()


# In[35]:

def get_user_count(data):
    return data.map(lambda r: int(r[0])).distinct().max()

user_count = get_user_count(ratings_rdd)
print 'Users: %d' % user_count


# In[36]:

from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix
from scipy.spatial import distance

vectorizer = DictVectorizer(sparse=True)

def vectorize(vals, user_count):
    vec = np.zeros((user_count+1, 1))
    for (u, r) in vals:
        vec[int(u)] = float(r)
    return vec

def get_k_similar2(movie, vector, movie_sparse_vectors, k):
#     print movie, movie_sparse_vectors
    dists = []
    for m, vec in movie_sparse_vectors.iteritems():
        if movie == m: continue
        dists += [(m, distance.euclidean(vectorize(vec, user_count), vectorize(vector, user_count)))]

    similar_movies = sorted(dists, key=lambda x: x[1])[:k]
    return similar_movies

def get_user_movies(data):
    user_movies = data.groupBy(lambda r: r[0]).mapValues(lambda r: np.array(list(r))[:, (1,2)])
    return dict(user_movies.collect())

def get_movie_features(data):
    movie_sparse_vectors = data.groupBy(lambda r: r[1]).mapValues(lambda r: np.array(list(r))[:, (0,2)])
    return dict(movie_sparse_vectors.collect())
    
def get_similar_rated_by_user(user, movie, user_movies, movie_sparse_vectors):
    user_rated_movies = user_movies[user][:, 0]
    user_ratings = dict(user_movies[user])
    
    movie_vecs = dict([(m, movie_sparse_vectors[m]) for m in user_rated_movies])
    
    cur_vec = movie_sparse_vectors[movie]
    similar = get_k_similar2(movie, cur_vec, movie_vecs, 10)
    return [(m, float(user_ratings[m]), sim) for m, sim in similar]


# In[37]:

# get mean for a grouped key result
def get_mean(r):
    return np.array(list(r))[:, 2].astype(float).mean()

# remove bias for one sample
def remove_sample_bias(r, user_means, movie_means, global_mean, user_movies, movie_sparse_vectors):
    user_mean = user_means[r[0]]
    baseline_offset = global_mean + (user_mean - global_mean) + (movie_means[r[1]] - global_mean)
#     r[2] =  round(min(max(r[2] + (user_mean - global_mean) + (movie_means[r[1]] - global_mean), 0), 5), 2)
    
    nn_offset = 0
#     sim_sum = 0
# #     print get_similar_rated_by_user(r[0], r[1], user_movies, movie_sparse_vectors)
#     for movie, rating, sim in get_similar_rated_by_user(r[0], r[1], user_movies, movie_sparse_vectors):
#         sim_sum += sim
#         baseline_sim = global_mean + (user_mean - global_mean) + (movie_means[movie] - global_mean)
#         nn_offset -= sim * (rating - baseline_sim)
        
# #         print sim, rating, baseline_sim, (rating - baseline_sim)
# #     print sim_sum
#     nn_offset /= sim_sum if sim_sum != 0 else 1
        
#     r[2] = round(min(max(r[2] + baseline_offset + nn_offset, 0), 5), 2) # bound and round rating
    r[2] -= baseline_offset
    return r

def add_sample_bias(r, user_means, movie_means, global_mean):
    user_mean = user_means[str(r[0][0])]
    baseline_offset = global_mean + (user_mean - global_mean) + (movie_means[str(r[0][1])] - global_mean)
    r = (r[0], r[1] + baseline_offset)
    return r

def remove_bias(data, global_mean, user_means, movie_means):
    print 'Getting user movies and movie features'
    user_movies = get_user_movies(data)
    movie_sparse_vectors = get_movie_features(data)
    
    print 'Removing bias'
    return data.map(lambda r: remove_sample_bias(r, user_means, movie_means, global_mean, user_movies, movie_sparse_vectors))

def add_bias(data, global_mean, user_means, movie_means):
    print 'Adding bias'
    return data.map(lambda r: add_sample_bias(r, user_means, movie_means, global_mean))


# In[38]:

print 'Getting means'
# global mean
global_mean = train_rdd.map(lambda r: r[2]).mean()
print global_mean

# user means
user_means = train_rdd.groupBy(lambda r: r[0]).mapValues(lambda r: get_mean(r)).collect()
user_means = dict(user_means)
# print user_means

# user means
movie_means = train_rdd.groupBy(lambda r: r[1]).mapValues(lambda r: get_mean(r)).collect()
movie_means = dict(movie_means)
# print movie_means

train_rdd_unbiased = remove_bias(train_rdd, global_mean, user_means, movie_means)
train_rdd_unbiased.collect()[:1]


# In[39]:

def train_als(data, rank, lambda_):
    # map ratings into Ratings object comprised of [user, movie, rating]
    data = data.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    model = ALS.train(data, rank=rank, seed=None, iterations=10, lambda_= lambda_)
    return model


# In[40]:

def evaluate(model, data, test=False, bias=True):
    #prepare data for predictions
    data=data.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    data_for_predict_RDD = data.map(lambda x: (x[0], x[1]))

    #find predictions
    predictions_data = model.predictAll(data_for_predict_RDD).map(lambda r: ((r[0], r[1]), r[2]))
    if bias:
        predictions_data = add_bias(predictions_data, global_mean, user_means, movie_means)

    #compute RMSE
    ratesAndPreds = data.map(lambda r: ((r[0], r[1]), r[2])).join(predictions_data)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error for " + str('test' if test else 'validation') + " = " + str(MSE))


# In[45]:

# for rank in [10, 20, 30, 40, 50]:
#     for lambda_ in [0.01, 0.1, 1.0, 10.0]:
rank = 20
lambda_ = 0.1
print rank, lambda_
print 'Training als'
model = train_als(train_rdd_unbiased, rank, lambda_)
print 'Evaluating'
evaluate(model, validation_rdd)


# In[16]:

print 'Training als'
model = train_als(train_rdd)
print 'Evaluating'
evaluate(model, validation_rdd, bias=False)


# In[ ]:



