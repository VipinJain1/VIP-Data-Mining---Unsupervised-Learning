# Data-Mining---Unsupervised-Learning

### Refer:
https://www.analyticsvidhya.com/blog/2015/12/10-machine-learning-algorithms-explained-army-soldier/
https://nicholastsmith.wordpress.com/2017/10/14/deep-learning-ocr-using-tensorflow-and-python/
https://github.com/chongyangtao/Awesome-Scene-Text-Recognition
https://cs.wmich.edu/alfuqaha/summer14/cs6530/lectures/ClusteringAnalysis.pdf  
http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering
http://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py  

 
PyClustering library also provide K-Medoid and many other clustering algorithms(Like DBSCAN, K-Means++ etc.).

LINK : https://pypi.org/project/pyclustering/

INSTALL : pip install pyclustering

DOCUMENTATION (Only K-Medoid) : 
https://codedocs.xyz/annoviko/pyclustering/classpyclustering_1_1cluster_1_1kmedoids_1_1kmedoids.html
 
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
Task: Find c1,c2....ck. We automatically find s1,s2,...sk.

Minimize the intra-cluster dist.

so
AvgMin for c1,c2,c3...ck  ( Sum (I to k all clusters) * Sum ( x:si)  ||x-ci|| square)) This is NP hard problem.
Whenever we have a hard problem, solve it for approximation. This is Lloyds’s algorithm.

### Lloyd's Algorithm:

(i) Initialization:
  Pick randomly k points from D - call them centroids.
(ii) Assignment:
 For each point Xi, select the nearest centroid Cj ; dist. (xi, Cj) J = 1,2....k.

(iii) Recompute centroid.
   Recalculate/update Cj's as follows:
    
    Cj = 1/|Sj| Sum (Xi) ; Xi all Sjs.
 (iv) Repeat step 2 and 3, until convergence - centroids do not change much. check old with new centroids. 

### How to initilize : K-Means+

Lloyd’s algo we did random initialization to pick k points.

Initialization sensitivity -> final cluster & centroids. Means how chosen initial data points. 

Refer:https://cs.wmich.edu/alfuqaha/summer14/cs6530/lectures/ClusteringAnalysis.pdf  

(i)  Get Original Points: -> generate clusters.
(ii) now optimal clustering. apply Lloyd ago. 
(iii) sub-optimal clustering. different initialization, apply Lloyd algo. 
you might get different results. Keep trying until you get stable centroids. 
so repeat k-means multiple tikes with different initialization and pick the best clustering based on smaller intra cluster and larger inter cluster distances.

### K-Means++:
Random initilization - smart initilization.
init in k-means: task pick c1,c2,....ck.
(i) Pick the first centroid c1 randomly from D dataset.
(ii) create a distribution as follows.
  for each Xi, xi = square of dist. (Xi, to the nearest centroid); distance square of each point to the centroid for each cluster and Create distances d1, d2, dn. pick a point from D-{C1} with a prob to di.

Trying to pick points that are as far as possible which has highest value of distance square of (X, nearest centroid).
K-Means++ does gert affected by outliers.. ifd we have iassue, repeat K-Means++ n times.

### Limitation of k-Means:

Clusters of different sizes, densities, non-globular shapes. k-mesa would not work.
Solution: increase K value, and put all together to make stable cluster, but hard job to do.  

### K-Medoids:
Problem K-Means I have centroids c1,c2...can. my centroids may not be interpretable.

Example Amazon Food review:
Each Xi represent review (review text) using BOW. my cells might have some value that might be hard to interpret. 
Instead of giving me Cj's centroid using means, if you gave me X10 part of D, as the first centroid can read my review text.
so, what if each centroid is each data point in data set 'D' so Ci = Xj :D
This algorithm is K-medoids. very popular algorithm.

Partitioning around Medoids (PAM):
(i) Same initialization like in Kmeans++  : pick k points from data set.
(ii) Assignment:  same thing, closed medoids. If xi : Sj if medoid j is the closest medoid 
(iii) Update/Recompute: Here is the difference.  we say, for each medoid, we 
 (a) swap each medoid point with non-medoid point if loss reduces go ahead otherwise swap again. 
 Example we have x1,x2,...x5,x6,x7...x10
 if I have x1 to x5 non medoid points and x6-x10 are medoid points, say swap x1 with x7.
 
 so, we are trying to minimize loss function using below.
 
 Minimize: Sum ( i to K) ||x-Mj||square.
 
  Example we have x1, x2, x5, x6, x7...x10
  a} compute loss function. Loss value x1=M1, x6 =m2. calculate Loss Value 1.
  b} swap M1=x2 and M2 =x6 compute the loss value again.
  c} lots of swaps are possible and keep recomputing loss value.
  d} reassign the points. 
  e} update the points.

### Determine the right K
1. From domain knowledge.
2. Elbow or Knee method.
 Loss function of K means - try to minimize sum of across all K clusters, distance between ||x-ci|| square.
We need to declare best K to calculate loss. if you draw the graph, between loss and k points, shape comes like elbow share.

###Time Complexity:

K-means: O(nKdi)  Linear time complexity.
Space: O(nd+Kd), order of N(d) linear.

## Hierarchical Clustering:
Reference:  
https://cs.wmich.edu/alfuqaha/summer14/cs6530/lectures/ClusteringAnalysis.pdf
http://scikit-learn.org/stable/modules/generated/sklearn.cluster.AgglomerativeClustering.html#sklearn.cluster.AgglomerativeClustering http://scikit-learn.org/stable/auto_examples/cluster/plot_agglomerative_clustering.html#sphx-glr-auto-examples-cluster-plot-agglomerative-clustering-py  

Two types:
 Agglomerative and Divisive Clustering.
 
### Agglomerative Clustering:

It assumes each point is a cluster to itself, it takes two cluster and group together and keep doing and group all together and finally make a big cluster for all the points that are near to each other. In last stage, we have only one big cluster.

The big challenge is whichever cluster is merged I need to refresh corresponding row and column on matrix. 

### Divisive:
It is just opposite of agglomerative. it tries to break large cluster into small clusters.
Problem is -How to divide?

#### How to define Inter-Cluster Similarity:

Min approach: when chose points pick points from cluster 1 and cluster 2 that have min distance.
Max approach: when chose points pick points from cluster 1 and cluster 2 that have max distance.
Group Avg: Take every pair point, compute the similarity and get the average. 
Distance between centroids.

#### Limitation of Hierarchical Clustering
1. No math objective for that, we are directly solving. K-mean - clear math objective. Hierarchical is a neat solution but does not fit into math solution.
2. Weather I use Min, Max or Avg, all have their own limitation. Min - has outliers’ issue. max - breaks large cluster. 
3. Most imp limitation: - cannot be used when N is large. Space and Time complexity. Space O (N2) and Time O (N2logn)
K-means has O(nd) complexity. 

### DBSCAN Density Based Clustering:

#### Measuring Density: Measure MinPts and Eps - both are hyper parameters.
They define how DBSCan works.
1. Density at point P defines as: number of points within a hypersphere of radius eps around p.
2. Dense region: a hypersphere of radius eps that contains at least minpts points.  if minpts is 4 and I have points 5, then sphere has dense region. if points are 3 and minpts =4 then it is sparse region.
#### Core point, Border point and noise point:

D = {Xi}, MinPts and Eps.

#### Core Point -P: if Point p has >= MinPts points in an eps radius around it. Core point 'P' always belongs to a dense region.
#### Border point:  Point P is set to be a border point if P is not a core point, that means P has <MinPts  points in eps radius.
 and P is part of neighborhood (Q - Core point)
 and dist. (P, Q) <= eps.

#### Noise Point: neither core or border point is called noise points.

 

 









  

  



 
    
 




















 




















