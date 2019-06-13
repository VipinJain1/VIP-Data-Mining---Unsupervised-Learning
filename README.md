# Data-Mining---Unsupervised-Learning
### Clustering:

What we learnt: Classfication and Regression problem.
D = {Xi, Yi}: data set
Yi : {0,1} -> 2-class classification problem
Yi : IR -> Regression
Y = F(x)
Clustring:
Given bunch of data points, no Yi's
Given is just D={Xi}
Task: group or cluster similar data points.
How do you measure how well my algorithm performed if Yi is not given?

Clustering (Unsupervised learning) ->No Yi's to supervise learning. 
Classification and regression - supervised learning algos since given Xi and Yi.

Semi-supervised learning:
D = D1 U D2.
D1 -> {Xi, Yi} -> Small problem.
D2= {Xi} ->Large problem

Applications of clustering: -> used in Data Mining. 
go to Wiki:
https://en.wikipedia.org/wiki/Cluster_analysis

Example used on eCommerce:
Amazon, Alibaba, Ebay, Flipkart.
Task: Group similar customers based on their purchasing behavior. how much money used, credit card, income limit, geo area.
if I group by purchasing behavior:
So If I create example 5 clusters based on purchasing behaviors and create deals.
deal -> C1, C2, C3, C4, C5.

These deals are created based on customer's behaviors. 

Second problem:
#### Image segmentation:

I can segment/regions image based on part. grouping/clustering similar pixels. to detect image.
Typically, ML algo to perform obj detection.

Example Amazon Food review:
-------------------------------------
Reviews are done manually. If I have millions of reviews, super expensive work if I need to label those reviews.
Steps:
I cluster my 1M reviews into cluster based on words in the reviews.
I create may be 10K clyster from 1M reviews based on words from reviews.
so instead of going to 1M review I just go 10k cluster.
some based on cluster I give review Positive or negative review and I label the product.

#### Metrics of Clustering:

How to measure how good is my cluster is since we have no Yi, only data set is given?

Intra-cluster -> within cluster.
Inter-cluster -> between cluster.

Keep intra-cluster distance small and Inter-cluster distance keep large, core idea of measuring clustering effectiveness.

#### Dunn- Index:
D : max D (i,j)/max D'(k) 
D-> distance between  cluster I and j  so max is max inter-cluster dist.
D' -> intra-cluster dist k.

### K-Means: Geometric Intuition, Centroids:
K -> number of clusters. each cluster has center point -> centroids.
3 clusters, so 3 Centroids, K=3
so total data set  = S1 U S2 U S3 if data is in 3 clusters named S1, S2 and S3.
so K- Clusters: 1,2,3....k
K-centroids: C1,c2,c3....Ck
K-Sets: S1,s2,S3..........Sk.
Ci = 1/n Sum (Xj) -> Xj is 1 to Sj.  so Ci is mean point of Si.
K-means is centroid based clustering.
Big-challange: How to find K-centeroids.
K-Centroids ->  K-sets nearest centroids.

#### K-Means : Mathematical Formulation:

D = {xq,x2,x3...xk}
Task: K-centroids :- c1,c2,c3....ck.
Sets : s1,s2,s3....sk.
Constraints:  Si Intersection Sj = 0, no common points in intersection and Xi is part of Sj. so every point has to be part of any cluster.
Task: Find c1,c2....ck. We automatically find S1,s2,...sk.


















 




















