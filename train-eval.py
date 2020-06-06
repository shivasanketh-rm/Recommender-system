from pyspark.mllib.evaluation import RegressionMetrics, RankingMetrics
from pyspark.mllib.recommendation import ALS, Rating
from pyspark.sql.functions import UserDefinedFunction, explode, desc
from pyspark.sql import SparkSession
from pyspark.sql.window import Window
import pyspark.sql.functions as F
from pyspark.sql.functions import col
import time
import sys

spark = SparkSession.builder.appName('abc').getOrCreate()

if len(sys.argv) == 4:
    TRAINING_SET, VALIDATION_SET, TEST_SET = sys.argv[1:]
else:
    print("Invalid number of    command line arguments")
    TRAINING_SET, VALIDATION_SET, TEST_SET = ['train_final_data.parquet', 'val_data.parquet', 'train_final_data.parquet']


def rate(x):
    '''
    Function to map as per MLlib ALS recommender sytem requirement
    x: type-RDD: input to be mapped

    Return
    ----------
    Mapped Ratings: type-RDD
    '''
    # Map as user, product, rating from user_id, book_id, rating
    return Rating(int(x[0]), int(x[1]), float(x[2]))


def model_train(spark, train_data, val_test, tval, iterations=15, rank=20, lambda_=0.1):
    '''
    Function to train and evaluate a recommender system

    Parameters
    ----------
    spark : spark session object
    train_data: type-RDD: Training data
    val_test: type-RDD: Validation or Testing data
    tval: type-RDD: Validation/Testing data mapped
    iterations: type-Int: Number of iterations to run
    rank: type-int: latent factors
    lambda_: type-int: learning rate

    Return
    ----------
    model: type-MLlib model: developed model
    scoreAndLabels: type-RDD: predicted scores and actual Labels
    '''
    # Record start time
    start_time = time.time()
    # Fit ALS model with the given parameters
    model = ALS.train(ratings=train_data, iterations=iterations, rank=rank, lambda_=lambda_)
    # Record Fit time
    fit_time = time.time()
    # Predict all Test/Val data
    predictions = model.predictAll(val_test).map(lambda r: ((r.user, r.product), r.rating))
    # Record Predict time
    predict_time = time.time()
    ratingsTuple = tval.map(lambda r: ((r.user, r.product), r.rating))
    # Create Scores and Labels tuples
    scoreAndLabels = predictions.join(ratingsTuple).map(lambda tup: tup[1])
    # calculate RMSE
    metrics = RegressionMetrics(scoreAndLabels)
    print("RMSE = %s" % metrics.rootMeanSquaredError)
    print("Time to fit = {} min".format((fit_time - start_time)/60))
    print("Time to predict = {} min".format((predict_time - fit_time)/60))
    return model, ratingsTuple, scoreAndLabels


def train_prep(spark, training_data, val_test_data):
    '''
    Function to prepare training and testing data

    Parameters
    ----------
    spark : spark session object
    training_data: type-Dataframe: Unprocessed training data
    val_test_data: type-Dataframe: Unprocessed val/test data

    Return
    ----------
    train_data: type-RDD: Training data
    val_test: type-RDD: Validation or Testing data
    tval: type-RDD: Validation/Testing data mapped

    '''
    # Map Train and Val/Test data
    train_data = training_data.rdd.map(lambda r: rate(r))
    tval = val_test_data.rdd.map(lambda r: rate(r))
    val_test = tval.map(lambda r: (r.user, r.product))
    return train_data, val_test, tval


def mAPandprecisionatK(spark, model, k, labels, user_ids):
    '''
    Function to print the metric meanAveragePrecision and Precisionatk

    Parameters
    ----------
    spark : spark session object
    model: type-MLlib model: developed model
    k: type-int: Top-k predictions for every user
    scoreAndLabels: type-RDD: predicted scores and actual Labels
    user_ids: user_ids to recommend products for

    return
    ----------
    None
    '''
    recs = []
    for uid in user_ids:
        # recommend k products for each user
        temp_recs = model.recommendProducts(uid.user_id, k)
        # collect only the book_ids from the recommendations
        recs.append([temp_rec.product for temp_rec in temp_recs])

    l = labels.map(lambda tup: float(tup[1])).collect()
    rdd = spark.sparkContext.parallelize([(recs, l)])
    m = RankingMetrics(rdd)
    print("meanAveragePrecision {}".format(m.meanAveragePrecision))
    print("Precision at K for K ={} is {}" .format(k, m.precisionAt(k)))


# Pyspark command terminal
# Read data
training_data = spark.read.parquet(TRAINING_SET)
validation_data = spark.read.parquet(VALIDATION_SET)
test_data = spark.read.parquet(TEST_SET)

val_user_ids = validation_data.select('user_id').distinct().collect()
test_user_ids = test_data.select('user_id').distinct().collect()

# Train and Val/Test data Preparation
train_data, val_test, tval = train_prep(spark, training_data, validation_data)
# Model Fitting and Evaluation
model, ratingsTuple, scoreAndLabels = model_train(spark, train_data, val_test, tval, iterations=15, rank=20, lambda_=0.01)

ratingsTuple = ratingsTuple.toDF()
# drop the rating in the ratingsTuple
ratingsTuple = ratingsTuple.select([F.col('_1._1').alias('user_id'), F.col('_1._1').alias('book_id')])
# group the ratingsTuple by user_id and collect the book recommendations for each user as a list
collectedLabels = ratingsTuple.groupBy('user_id').agg(F.collect_list('book_id')).collect()
# each element in collectedLabels is object of Row, so remove the attribute names and make tuples of type (user_id, [book_ids])
collectedLabels = [(row.user_id, row['collect_list(book_id)']) for row in collectedLabels]

# Print Precision at k for validation data
mAPandprecisionatK(spark, model, 500, collectedLabels, val_user_ids)
