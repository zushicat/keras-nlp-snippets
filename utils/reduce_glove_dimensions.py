# From:
# https://github.com/vyraun/Half-Size

import numpy as np
from sklearn.decomposition import PCA

Glove = {}
f = open("data/glove/german_vectors_100.txt")

print("Loading Glove vectors.")
for line in f:
    values = line.split()
    word = values[0]
    coefs = np.asarray(values[1:], dtype='float32')
    Glove[word] = coefs
f.close()

print("Done.")
X_train = []
X_train_names = []
for x in Glove:
        X_train.append(Glove[x])
        X_train_names.append(x)

X_train = np.asarray(X_train)
pca_embeddings = {}

# PCA to get Top Components
pca =  PCA(n_components = 100)
X_train = X_train - np.mean(X_train)
X_fit = pca.fit_transform(X_train)
U1 = pca.components_

z = []

# Removing Projections on Top Components
for i, x in enumerate(X_train):
	for u in U1[0:7]:        
        	x = x - np.dot(u.transpose(),x) * u 
	z.append(x)

z = np.asarray(z)

# PCA Dim Reduction
pca =  PCA(n_components = 50)
X_train = z - np.mean(z)
X_new_final = pca.fit_transform(X_train)


# PCA to do Post-Processing Again
pca =  PCA(n_components = 50)
X_new = X_new_final - np.mean(X_new_final)
X_new = pca.fit_transform(X_new)
Ufit = pca.components_

X_new_final = X_new_final - np.mean(X_new_final)

final_pca_embeddings = {}
with open("data/glove/german_vectors_50.txt", "w") as f:
        for i, x in enumerate(X_train_names):
                final_pca_embeddings[x] = X_new_final[i]
                f.write(f"{x}\t")
                
                for u in Ufit[0:7]:
                        final_pca_embeddings[x] = final_pca_embeddings[x] - np.dot(u.transpose(),final_pca_embeddings[x]) * u 

                for t in final_pca_embeddings[x]:
                        f.write(f"{t}\t")
                
                f.write("\n")

