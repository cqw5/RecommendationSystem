### [Surprise(Simple Python RecommendatIon System Engine)](https://github.com/NicolasHug/Surprise)

一个Python的推荐系统库，Scikit系列产品

### 推荐算法：

* [基础算法/baseline algorithms](http://surprise.readthedocs.io/en/stable/basic_algorithms.html)
* [基于近邻方法(协同过滤)/neighborhood methods](http://surprise.readthedocs.io/en/stable/knn_inspired.html)
* [矩阵分解方法/matrix factorization-based (SVD, PMF, SVD++, NMF)](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)

| 算法类名                                     | 说明                                       |
| ---------------------------------------- | :--------------------------------------- |
| [random_pred.NormalPredictor](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.random_pred.NormalPredictor) | Algorithm predicting a random rating based on the distribution of the training set, which is assumed to be normal. |
| [baseline_only.BaselineOnly](http://surprise.readthedocs.io/en/stable/basic_algorithms.html#surprise.prediction_algorithms.baseline_only.BaselineOnly) | Algorithm predicting the baseline estimate for given user and item. |
| [knns.KNNBasic](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBasic) | A basic collaborative filtering algorithm. |
| [knns.KNNWithMeans](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNWithMeans) | A basic collaborative filtering algorithm, taking into account the mean ratings of each user. |
| [knns.KNNBaseline](http://surprise.readthedocs.io/en/stable/knn_inspired.html#surprise.prediction_algorithms.knns.KNNBaseline) | A basic collaborative filtering algorithm taking into account a baseline rating. |
| [matrix_factorization.SVD](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD) | The famous SVD algorithm, as popularized by Simon Funk during the Netflix Prize. |
| [matrix_factorization.SVDpp](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVDpp) | The SVD++ algorithm, an extension of SVD taking into account implicit ratings. |
| [matrix_factorization.NMF](http://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.NMF) | A collaborative filtering algorithm based on Non-negative Matrix Factorization. |
| [slope_one.SlopeOne](http://surprise.readthedocs.io/en/stable/slope_one.html#surprise.prediction_algorithms.slope_one.SlopeOne) | A simple yet accurate collaborative filtering algorithm. |
| [co_clustering.CoClustering](http://surprise.readthedocs.io/en/stable/co_clustering.html#surprise.prediction_algorithms.co_clustering.CoClustering) | A collaborative filtering algorithm based on co-clustering. |



### 距离度量准则：

| 相似度度量标准                                  | 度量标准说明                                   |
| ---------------------------------------- | :--------------------------------------- |
| [cosine](http://surprise.readthedocs.io/en/stable/similarities.html#surprise.similarities.cosine) | Compute the cosine similarity between all pairs of users (or items). |
| [msd](http://surprise.readthedocs.io/en/stable/similarities.html#surprise.similarities.msd) | Compute the Mean Squared Difference similarity between all pairs of users (or items). |
| [pearson](http://surprise.readthedocs.io/en/stable/similarities.html#surprise.similarities.pearson) | Compute the Pearson correlation coefficient between all pairs of users (or items). |
| [pearson_baseline](http://surprise.readthedocs.io/en/stable/similarities.html#surprise.similarities.pearson_baseline) | Compute the (shrunk) Pearson correlation coefficient between all pairs of users (or items) using baselines for centering instead of means. |



### 评估准则
| 评估准则                                     | 准则说明                                     |
| ---------------------------------------- | :--------------------------------------- |
| [rmse](http://surprise.readthedocs.io/en/stable/accuracy.html#surprise.accuracy.rmse) | Compute RMSE (Root Mean Squared Error).  |
| [msd](http://surprise.readthedocs.io/en/stable/similarities.html#surprise.similarities.msd) | Compute MAE (Mean Absolute Error).       |
| [fcp](http://surprise.readthedocs.io/en/stable/accuracy.html#surprise.accuracy.fcp) | Compute FCP (Fraction of Concordant Pairs). |

