import numpy as np
from sklearn import preprocessing
from lightfm import LightFM
from scipy.sparse import csr_matrix 
from scipy.sparse import coo_matrix
import time
from lightfm.evaluation import auc_score
from lightfm.evaluation import precision_at_k



def single_machine_model(spark, training_data, validation_data, test_data):
    '''
    Function to preprocess data and create sparse matrix

    Parameters
    ----------
    spark : spark session object
    training_data: type-Dataframe: Unprocessed training data
    validation_data:type-Dataframe: Unprocessed validation data
    test_data: type-Dataframe: Unprocessed testing data 

    Return
    ----------
    train: Type-Sparse Matrix: Processed training data
    val: Type-Sparse Matrix: Processed validation data
    test: Type-Sparse Matrix: Processed testing data

    '''
    #Create Views
    training_data.createOrReplaceTempView('training_data')
    validation_data.createOrReplaceTempView('validation_data')
    test_data.createOrReplaceTempView('test_data')
    #Find all the unique books in training
    Unique_book_id_list = spark.sql("SELECT DISTINCT(book_id) FROM training_data")
    Unique_book_id_list.createOrReplaceTempView('Unique_book_id_list')
    
    #Update val and test set accordingly by eliminating book ids that are present in Validation/Testing but not in Training
    val_set_updated = spark.sql(\
        "SELECT val_final.user_id, val_final.book_id, val_final.rating  FROM val_final Inner join Unique_book_id_list on val_final.book_id = Unique_book_id_list.book_id")
    test_set_updated = spark.sql\
        ("SELECT test_final.user_id, test_final.book_id, test_final.rating  FROM test_final Inner join Unique_book_id_list on test_final.book_id = Unique_book_id_list.book_id")

    #Take User ids
    train_user_id = np.array(training_data.select('user_id').collect())
    val_user_id = np.array(val_set_updated.select('user_id').collect())
    test_user_id = np.array(test_set_updated.select('user_id').collect())

    #Take Book ids
    train_book_id = np.array(training_data.select('book_id').collect())
    val_book_id = np.array(val_set_updated.select('book_id').collect())
    test_book_id = np.array(test_set_updated.select('book_id').collect())

    #Take ratings
    train_rating = np.array(training_data.select('rating').collect())
    val_rating = np.array(val_set_updated.select('rating').collect())
    test_rating = np.array(test_set_updated.select('rating').collect())

    #Reshaping as required by preprocessing.LabelEncoder()
    train_user_id = np.reshape(train_user_id, (train_user_id.shape[0]))
    val_user_id = np.reshape(val_user_id, (val_user_id.shape[0]))
    test_user_id = np.reshape(test_user_id, (test_user_id.shape[0]))

    #Reshaping as required by preprocessing.LabelEncoder()
    train_book_id = np.reshape(train_book_id, (train_book_id.shape[0]))
    val_book_id = np.reshape(val_book_id, (val_book_id.shape[0]))
    test_book_id = np.reshape(test_book_id, (test_book_id.shape[0]))

    #Reshaping as required by preprocessing.LabelEncoder()
    train_rating = np.reshape(train_rating, (train_rating.shape[0]))
    val_rating = np.reshape(val_rating, (val_rating.shape[0]))
    test_rating = np.reshape(test_rating, (test_rating.shape[0]))

    #Creation of Sparse Matrix
    trans_cat_train = dict()
    trans_cat_val = dict()
    trans_cat_test = dict()

    cate_enc = preprocessing.LabelEncoder()
    trans_cat_train['user_id'] = cate_enc.fit_transform(train_user_id)
    trans_cat_val['user_id'] = cate_enc.transform(val_user_id)
    trans_cat_test['user_id'] = cate_enc.transform(test_user_id)

    cate_enc = preprocessing.LabelEncoder()
    trans_cat_train['book_id'] = cate_enc.fit_transform(train_book_id)
    trans_cat_val['book_id'] = cate_enc.transform(val_book_id)
    trans_cat_test['book_id'] = cate_enc.transform(test_book_id)


    rating_dict = dict()
    cate_enc = preprocessing.LabelEncoder()
    rating_dict['train'] = cate_enc.fit_transform(train_rating)
    rating_dict['val'] = cate_enc.transform(val_rating)
    rating_dict['test'] = cate_enc.transform(test_rating)

    n_users = len(np.unique(trans_cat_train['user_id']))
    n_items = len(np.unique(trans_cat_train['book_id']))

    #Encoding of Sparse Matrix
    train = coo_matrix((rating_dict['train'], (trans_cat_train['user_id'],trans_cat_train['book_id'])), shape=(n_users, n_items))
    val = coo_matrix((rating_dict['val'], (trans_cat_val['user_id'],trans_cat_val['book_id'])), shape=(n_users, n_items))
    test = coo_matrix((rating_dict['test'], (trans_cat_test['user_id'],trans_cat_test['book_id'])), shape=(n_users, n_items))

    return train, val, test

def model_train_test(spark, train, val, test, no_components = 20, learning_rate = 0.01, epochs = 15, k=10):
    '''
    Function to train Lightfm collaborative filtering recommender system and test on validation data and testing data

    Parameters
    ----------
    spark : spark session object
    train: Type-Sparse Matrix: Processed training data
    val: Type-Sparse Matrix: Processed validation data
    test: Type-Sparse Matrix: Processed testing data
    No_components: type-Int: latent factors
    learning_rate: type-int: learning rate
    epochs: type-int: Iterations
    k: type-int: Top-k predictions for every user

    Return
    ----------
    None
    '''
    start_time = time.time()
    #Create Lightfm object   
    model= LightFM(no_components=no_components,learning_rate=learning_rate,loss='warp')
    #Fit model
    model.fit(train,epochs=epochs,num_threads=1)
    #Record time (Fit time)
    fit_time = time.time()
    #Calculate AUC value
    auc_val = auc_score(model, val).mean()
    score_calc_time = time.time()
    auc_test = auc_score(model, test).mean()
    auc_train = auc_score(model, train).mean()

    #Calculate Precision_at_k
    P_at_K = precision_at_k(model,test, k=k )
    precision_value = np.mean(P_at_K)
    print("For no_components = {}, learning_rate = {} " .format(no_components,learning_rate ))
    print("Train AUC Score: {}".format(auc_train))
    print("Val AUC Score: {}".format(auc_val))
    print("Test AUC Score: {}".format(auc_test))
    print("Precision at k={} Score: {}".format(k, precision_value))
    print("--- Fit time:  {} mins ---".format(fit_time - start_time))
    print("--- Score time:  {} mins ---".format(score_calc_time - fit_time))


#Pyspark command terminal
#Read data
training_data = spark.read.parquet('train_final_data.parquet')
validation_data = spark.read.parquet('val_data.parquet')
test_data = spark.read.parquet('train_final_data.parquet')

train, val, test = single_machine_model(training_data, validation_data, test_data)
model_train_test(spark, train, val, test, no_components = 20, learning_rate = 0.01, epochs = 15, k = 10)
