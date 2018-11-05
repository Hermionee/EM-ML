import numpy as np
import scipy.stats as stats
import csv
import pandas as pd
from scipy import sparse
import matplotlib.pyplot as plt
import numpy.linalg as linear
d = 5
T = 100
sigma=1
epsilon=1e-7 # this parameter is to prevent tiny probability leading to log invalid
c=1
URV = pd.read_csv('./movies_csv/ratings.csv').values
URV_test=pd.read_csv('./movies_csv/ratings_test.csv').values
with open('./movies_csv/movies.txt') as csv_file:
    csv_reader = csv.reader(csv_file, delimiter=',')
    movies = 0
    for row in csv_reader:
        movies += 1
    M = movies

users = []
r = np.array([])
col = np.array([])
data = np.array([], dtype=float)

r = np.array([])
col = np.array([])
data = np.array([], dtype=float)

for row in URV:
    if row[0] not in users:
        r = np.append(r, len(users))
        users.append(row[0])
    else:
        r = np.append(r, users.index(row[0]))
    col = np.append(col, int(row[1]) - 1)
    data = np.append(data, int(row[2]))

N = len(users)

R = sparse.coo_matrix((data, (r, col)), shape=(N, M))


r = np.array([])
col = np.array([])
data = np.array([], dtype=float)
for row in URV_test:
    r = np.append(r, users.index(row[0]))
    col = np.append(col, int(row[1]) - 1)
    data = np.append(data, int(row[2]))

R_test = sparse.coo_matrix((data, (r, col)), shape=(N, M))

ones = np.nonzero(R == 1)
nones = np.nonzero(R == -1)

def EM(U,V):
    uv = np.dot(U, V.transpose()) / sigma
    results=[]
    for i in range(T):
        phi=np.zeros((N,M))
        phi[ones]=uv[ones]+sigma*stats.norm.pdf(uv[ones]/sigma, 0, 1) / (stats.norm.cdf(uv[ones]/sigma,0,1)+epsilon)
        phi[nones]=uv[nones]-sigma*stats.norm.pdf(uv[nones]/sigma, 0, 1) / (1 - stats.norm.cdf(uv[nones]/sigma, 0, 1)+epsilon)
        if i%2==0:
            e = np.identity(d)/c
            for vj in V:
                e =np.add(e,np.outer(vj,vj)/(sigma*sigma))
            U = np.dot(linear.inv(e),phi.dot(V).transpose()/(sigma*sigma)).transpose()
        else:
            e = np.identity(d)/c
            for uj in U:
                e = np.add(e,np.outer(uj,uj)/(sigma*sigma))
            V = np.dot(linear.inv(e),U.transpose().dot(phi)/(sigma*sigma)).transpose()


        uv = np.dot(U, V.transpose())/sigma
        w=np.sum(np.log(stats.norm.cdf(uv[ones]/sigma,0,1)+epsilon))
        z=np.sum(np.log(1-stats.norm.cdf(uv[nones]/sigma,0,1)+epsilon))
        y=w+z
        print(y)
        results.append(y)



    plt.plot(np.arange(T)[19:],results[19:])
    return U,V



if __name__=='__main__':
    # print("Problem 1")
    # #INITIALIZE U V
    # mean=np.zeros(d)
    # cov=0.1*np.identity(d)
    # U = np.random.multivariate_normal(mean, cov, N)
    # V = np.random.multivariate_normal(mean, cov, M)
    #
    # EM(U,V)
    # plt.show()
    # print("Problem 2")
    # for i in range(5):
    #     results = []
    #
    #     mean=np.zeros(d)
    #     cov=0.1*np.identity(d)
    #     U = np.random.multivariate_normal(mean, cov, N)
    #     V = np.random.multivariate_normal(mean, cov, M)
    #     ui = np.sqrt(np.sum(U ** 2, axis=1))
    #     U /= (ui[:, np.newaxis])
    #     vj = np.sqrt(np.sum(V ** 2, axis=1))
    #     V /= (vj[:, np.newaxis])
    #     EM(U,V)
    #
    # plt.show()


    print("Problem 3")
    results=[]
    mean = np.zeros(d)
    cov = 0.1 * np.identity(d)
    U = np.random.multivariate_normal(mean, cov, N)
    V = np.random.multivariate_normal(mean, cov, M)
    U,V=EM(U,V)

    p=stats.norm.cdf((U.dot(V.transpose()))[np.nonzero(R_test!=0)]/sigma,0,1)
    R_test=R_test.toarray()
    R_test=R_test[np.nonzero(R_test!=0)]
    R_result=np.zeros(R_test.shape)
    positive=np.nonzero(p[np.nonzero(p!=0)]>=0.5)
    negative=np.nonzero(p[np.nonzero(p!=0)]<0.5)

    R_result[positive]=np.ones(positive[0].shape)
    R_result[negative]=-np.ones(negative[0].shape)
    print(np.mean(R_result==R_test))

    print(np.sum(R_result[positive] == R_test[positive]))
    print(np.sum(R_result[positive] != R_test[positive]))
    print(np.sum(R_result[negative] == R_test[negative]))
    print(np.sum(R_result[negative] != R_test[negative]))