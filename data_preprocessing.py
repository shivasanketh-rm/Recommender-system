#!/usr/bin/env python
# -*- coding: utf-8 -*-

from pyspark.sql import SparkSession
from pyspark.sql.functions import percent_rank
from pyspark.sql.functions import col
import pyspark.sql.functions as F
from pyspark.sql.window import Window



def data_split(spark, sample_size = 0.1, seed = 1):
    '''
    Function to Data Splitting and Sub sampling
    
    Parameters
    ----------
    spark : spark session object
    sample_size: type-float: Data subsampling 1 for complete data 
    seed: Seed for sampling
    
    Return
    ----------
    train_final_data: type-Dataframe:Training data
    val_data: type-Dataframe: Validation data
    test_data: type-Dataframe: Testing data
    '''

    #Load Dataset
    interactions = spark.read.csv('hdfs:/user/bm106/pub/goodreads/goodreads_interactions.csv',\
                                header=True, schema="user_id INT, book_id INT,is_read INT,rating INT,is_reviewed INT")
    #Create temp view 
    interactions.createOrReplaceTempView('interactions')

    #Remove Rows with is_read = 0, is_reviewed = 0 and ratings = 0
    filtered_interactions = spark.sql(
        "SELECT * FROM interactions WHERE is_read=1 AND is_reviewed=1 AND rating > 0")
    filtered_interactions.createOrReplaceTempView('filtered_interactions')

    # keep data from 10 or more interactions per user
    more_than_ten_interactions = spark.sql("SELECT D.book_id, D.is_read, D.rating, D.user_id, D.is_reviewed from filtered_interactions D INNER JOIN \
        (SELECT user_id, count(user_id) FROM filtered_interactions GROUP BY user_id having count(user_id) > 10) R ON D.user_id = R.user_id")
    more_than_ten_interactions.createOrReplaceTempView('more_than_ten_interactions')

    #Get all the unique user ids from the generated dataframe
    user_id_sampled = more_than_ten_interactions.select(more_than_ten_interactions.user_id).distinct()
    #Sub sampling
    if sample_size < 1:
        #Get-sub sampled unique user id list
        user_id_sampled = user_id_sampled.sample(withReplacement = False, fraction = sample_size, seed = seed)
        #Get interactions corresponding to sub-sampled unique user id list
        more_than_ten_interactions = more_than_ten_interactions.join(user_id_sampled, more_than_ten_interactions.user_id == user_id_sampled.user_id,'inner')\
                                .select(more_than_ten_interactions.user_id, more_than_ten_interactions.book_id,more_than_ten_interactions.rating )


    #Split unique user id  data in the ratio Train:0.6, Val:0.2, Test:0.2
    train_users, val_users, test_users = user_id_sampled.randomSplit([0.6, 0.2, 0.2])

    #Get corresponding Train/ Val/Test Interactions
    train_interactions = more_than_ten_interactions.join(train_users, more_than_ten_interactions.user_id == train_users.user_id,'inner')\
        .select(more_than_ten_interactions.user_id, more_than_ten_interactions.book_id,more_than_ten_interactions.rating )
    val_interactions = more_than_ten_interactions.join(val_users, more_than_ten_interactions.user_id == val_users.user_id,'inner')\
        .select(more_than_ten_interactions.user_id, more_than_ten_interactions.book_id,more_than_ten_interactions.rating )
    test_interactions = more_than_ten_interactions.join(test_users, more_than_ten_interactions.user_id == test_users.user_id,'inner')\
        .select(more_than_ten_interactions.user_id, more_than_ten_interactions.book_id,more_than_ten_interactions.rating )

    #Grouping
    val_interactions = (val_interactions.select('user_id','book_id','rating',percent_rank().over(Window.partitionBy(val_interactions['user_id']).orderBy(val_interactions['book_id'])).alias('percent_50')))
    #reserve 50 percent into training 
    val_to_train = val_interactions.filter(col('percent_50') < 0.5).select('user_id','book_id','rating')
    #Take remaining 50 percent as validation
    val_data = val_interactions.filter(col('percent_50') >= 0.5).select('user_id','book_id','rating')

    #Grouping
    test_interactions = (test_interactions.select('user_id','book_id','rating',percent_rank().over(Window.partitionBy(test_interactions['user_id']).orderBy(val_interactions['book_id'])).alias('percent_50')))
    #reserve 50 percent into training 
    test_to_train = test_interactions.filter(col('percent_50') < 0.5).select('user_id','book_id','rating')
    #Take remaining 50 percent as Testing
    test_data = test_interactions.filter(col('percent_50') >= 0.5).select('user_id','book_id','rating')

    #Append validation extracted rows into training
    train_val_data = train_interactions.unionByName(val_to_train)
    #Append Testing extracted rows into training
    train_final_data = train_val_data.unionByName(test_to_train)

    return train_final_data, val_data, test_data


#Pyspark command Terminal
train_final_data, val_data, test_data = data_split(spark, sample_size = 0.1, seed = 1)
#Write processed data
train_final_data.write.parquet('train_final_data.parquet')
val_data.write.parquet('val_data.parquet')
test_data.write.parquet('test_data.parquet')

