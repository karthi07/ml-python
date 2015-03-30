import os
import scipy as sp
import sys
from sklearn.feature_extraction.text import CountVectorizer,TfidfVectorizer


#Counting word and finding similarity

#Vectorizer with NLTK stemmer

import nltk.stem

english_stemmer = nltk.stem.SnowballStemmer('english')

class StemmedCountVectorizer(CountVectorizer):
  def build_analyzer(self):
    analyzer = super(StemmedCountVectorizer,self).build_analyzer()
    return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))


class StemmedTfidfVectorizer(TfidfVectorizer):
  def build_analyzer(self):
    analyzer = super(TfidfVectorizer,self).build_analyzer()
    return lambda doc: (english_stemmer.stem(w) for w in analyzer(doc))

#vectorizer = StemmedCountVectorizer(min_df=1,stop_words='english')

vectorizer = StemmedTfidfVectorizer(min_df=1,stop_words='english')

DIR = "text"
posts = [open(os.path.join(DIR,f)).read() for f in os.listdir(DIR)]
#print posts

X_train = vectorizer.fit_transform(posts)
num_samples,num_features = X_train.shape
#print vectorizer.get_feature_names()

new_post = "imaging databases"
new_post_vec = vectorizer.transform([new_post])
#print (new_post_vec.toarray())

def dist_raw(v1,v2):
  normalized_v1 = v1/sp.linalg.norm(v1.toarray())
  normalized_v2 = v2/sp.linalg.norm(v2.toarray())
  delta = normalized_v1-normalized_v2
  return sp.linalg.norm(delta.toarray())


best_doc = None
best_dist = sys.maxint
best_i = None

for i in range(0,num_samples):
  post = posts[i]
  if post == new_post:
    continue 
  post_vec = X_train.getrow(i)
  d = dist_raw(post_vec,new_post_vec)
  print "## Post %i with dist %.2f : %s"%(i,d,post)
  if d < best_dist:
    best_dist = d
    best_i = i

print "Best post is %i with dist %.2f"%(best_i,best_dist)


