import os
import scipy as sp
import sys
from sklearn.feature_extraction.text import CountVectorizer

#Vectorizing raw text 
#form text to bag of words

vectorizer = CountVectorizer(min_df=1,stop_words='english')
#print vectorizer

content = ["How to format my hard disk","Hard Disk format problems"]
X = vectorizer.fit_transform(content)
#print vectorizer.get_feature_names()
#print X.toarray().transpose()

#Counting word and finding similarity

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


