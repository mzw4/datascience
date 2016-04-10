
# coding: utf-8

# In[3]:

import sys, operator, time, os
from collections import defaultdict
import numpy as np

from pyspark import SparkContext
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating

# sc = SparkContext(appName="PythonPi")

# %matplotlib inline
# print os.environ["PYSPARK_SUBMIT_ARGS"]
# print sc._conf.get('spark.driver.memory')
# print sc._conf.get('spark.executor.memory')
# print sc._conf.getAll()

base_dir = 'ml-10M100K/'
ratings_dir = base_dir + 'ratings.dat'
movies_dir = base_dir + 'movies.dat'
tags_dir = base_dir + 'tags.dat'

test = False


# In[4]:

# load data as rdd
ratings_rdd = sc.textFile(ratings_dir)    .map(lambda r: [float(e) if i == 2 else e for i, e in enumerate(r.split('::'))])
movies_rdd = sc.textFile(movies_dir).map(lambda r: r.split('::'))
tags_rdd = sc.textFile(tags_dir).map(lambda r: r.split('::'))


# In[5]:

# for testing
if test:
    ratings_rdd = ratings_rdd.sample(False, 0.01, int(time.time()))


# In[6]:

def get_user_count(data):
    return data.map(lambda r: int(r[0])).distinct().max()

user_count = get_user_count(ratings_rdd)

print 'Users: %d' % user_count


# In[7]:

from sklearn.feature_extraction import DictVectorizer
from scipy.sparse import csr_matrix
from scipy.spatial import distance

vectorizer = DictVectorizer(sparse=True)

def vectorize(vals, user_count):
    vec = np.zeros((user_count+1, 1))
    for (u, r) in vals:
        vec[int(u)] = float(r)
    return vec
    
# def get_k_similar(vector, data, k):
#     distances = data.map(lambda r: (r[0], distance.euclidean(vectorize(r[1], user_count), vectorize(vector, user_count))))\
#         .sortBy(lambda d: d[1])
#     return distances.map(lambda m: m[0]).take(k)[1:]
      
# # get nearest k movie neighbors
# # use euclidean distance
# def get_nn_movies(ratings_data, k):
#     movie_sparse_vectors = ratings_data.groupBy(lambda r: r[1]).mapValues(lambda r: np.array(list(r))[:, (0,2)])
#     nn = {}
#     for movie, movie_vec in movie_sparse_vectors.collect():
#         print movie
#         nn[movie] = get_k_similar(movie_vec, movie_sparse_vectors, k)
#     return nn

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
    return [(m, sim, float(user_ratings[m])) for m, sim in similar]

#     nearest_neighbors = movie_sparse_vectors.map(lambda r: get_k_similar(r[1], movie_sparse_vectors, k))
#     return dict(nearest_neighbors.collect())


# In[8]:

user_movies = get_user_movies(ratings_rdd)
movie_sparse_vectors = get_movie_features(ratings_rdd)
# print user_movies['1'][:, 0]


# In[9]:

# from sklearn.feature_extraction import DictVectorizer
# from scipy.sparse import csr_matrix

# vectorizer = DictVectorizer(sparse=True)

# data = test_rdd.groupBy(lambda r: r[1]).mapValues(lambda r: np.array(list(r))[:, (0,2)])
# # print test_rdd.groupBy(lambda r: r[1]).mapValues(lambda r: np.array(list(r))[:, (0,2)]).collect()
# d = test_rdd.groupBy(lambda r: r[1]).mapValues(lambda r: np.array(list(r))[:, (0,2)]).collect()[0][1]
# d2 = test_rdd.groupBy(lambda r: r[1]).mapValues(lambda r: np.array(list(r))[:, (0,2)]).collect()[1][1]

# sm = vectorize(d, user_count)
# sm2 = vectorize(d2, user_count)
# # print sm.shape, sm2.shape
# print distance.cosine(sm, sm2)
# print d
# get_k_similar(d, data, 10)


# In[10]:

# get mean for a grouped key result
def get_mean(r):
    return np.array(list(r))[:, 2].astype(float).mean()

# remove bias for one sample
def remove_sample_bias(r, user_means, movie_means, global_mean, user_movies):
    user_mean = user_means[r[0]]
    baseline_offset = (global_mean - user_mean) + (global_mean - movie_means[r[1]])
    
    nn_offset = 0
    sim_sum = 0
    for movie, rating, sim in get_similar_rated_by_user(r[0], r[1], user_movies, movie_sparse_vectors):
        sim_sum += sim
        baseline_sim = global_mean + (user_mean - global_mean) + (movie_means[movie] - global_mean)
        nn_offset -= sim * (rating - baseline_sim)
    nn_offset /= sim_sum if sim_sum != 0 else 1
        
    r[2] = round(min(max(r[2] + baseline_offset + nn_offset, 0), 5), 2) # bound and round rating
    return r

def remove_bias(data):
    print 'Getting user movies'
    
    print 'Getting means'
    # global mean
    global_mean = data.map(lambda r: r[2]).mean()
    print global_mean
    
    # user means
    user_means = data.groupBy(lambda r: r[0]).mapValues(lambda r: get_mean(r)).collect()
    user_means = dict(user_means)
    # print user_means

    # user means
    movie_means = data.groupBy(lambda r: r[1]).mapValues(lambda r: get_mean(r)).collect()
    movie_means = dict(movie_means)
    # print movie_means
    
    print 'Removing bias'
    print data.take(1)[0]
#     print remove_sample_bias(data.take(1)[0], user_means, movie_means, global_mean, user_movies)
    
    return data.map(lambda r: remove_sample_bias(r, user_means, movie_means, global_mean, user_movies))


# In[ ]:

ratings_rdd_unbiased = remove_bias(ratings_rdd)


# In[233]:

def train_als(data):
    # map ratings into Ratings object comprised of [user, movie, rating]
    data = data.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    rank = 10
    numIterations = 10
    model = ALS.train(data, rank, numIterations)
    return model, data


# In[234]:

def evaluate(data, model):
    # Evaluate the model on training data
    testdata = data.map(lambda p: (p[0], p[1]))

    predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    ratesAndPreds = data.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    print("Mean Squared Error = " + str(MSE))


# In[11]:

print 'Training als'
model, data = train_als(ratings_rdd_unbiased)
print 'Evaluating'
evaluate(data, model)


# In[ ]:

print 'Training als'
model, data = train_als(ratings_rdd)
print 'Evaluating'
evaluate(data, model)

