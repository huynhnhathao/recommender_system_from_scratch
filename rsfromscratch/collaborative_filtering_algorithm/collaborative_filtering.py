
##################################################################
#                    Huynh Nhat Hao                              #
#         https://www.linkedin.com/in/haosleeper/                #
##################################################################

import logging
import sys
import copy
import numpy as np
from numpy.lib.financial import rate
import pandas as pd

log_format = '%(asctime)s %(message)s'
logging.basicConfig(stream=sys.stdout, level=logging.INFO,
                    format=log_format, datefmt='%m/%d %I:%M:%S %p')
logger = logging.getLogger()


class CollaborativeFiltering(object):
    """
    Base class for User-based and Item-based collaborative filtering

    methods: compute user mean rating, Pearson correlation between 2 array, 
    consine similarity between 2 arrays, centering the ratings accross user,

    """
    def __init__(self,data:pd.DataFrame, k_neighbors:int= 10, rating_matrix_row:str=None,
                 rating_matrix_column:str = None, rating_matrix_value:str = None, movies_data = None ) -> None:
        """
        Base class for Collaborative Filtering Algorithms
        """

        assert isinstance(data, pd.DataFrame), 'data must be a pandas dataframe'
        assert rating_matrix_row in data.columns, 'row must be a column in data'
        assert rating_matrix_column in data.columns, 'column must be a column in data'
        assert rating_matrix_value in data.columns, 'value must be a column in data'
        #create rating matrix from data
        self.rating_matrix = data.pivot( rating_matrix_row, rating_matrix_column, rating_matrix_value)
        self.k_neighbors = k_neighbors
        self.movies_data = movies_data
        # compute mean rating for each user
        self.user_mean_ratings = self.compute_user_mean_ratings()

        # a dict to save all the computed similarity score for all users
        self.all_similarity_score = dict.fromkeys(self.rating_matrix.columns, None)

        if movies_data is not None:
            self.movies_data = self.save_movies(movies_data)

        # replace nan with 0
        self.rating_matrix = self.rating_matrix.fillna(0)    

    def predict_rating(self, ) -> float:
        raise NotImplementedError

    def recommend(self,) -> list:
        raise NotImplementedError

    @classmethod
    def save_movies(cls, movie_data:pd.DataFrame ) -> None:
        """
        Read all movies and its id
        """
        return {id:title for i, (id, title, genres) in movie_data.iterrows()}

    def get_movie_names(self, ids:list) -> list:
        """
        Get the movie names from the previous saved movies

        param:
            ids: a list contains the id of the movies

        return a list of movie names of the provided ids
        """

    def compute_user_mean_ratings(self) -> dict:
        """
        compute and save the mean rating for each user in the rating matrix
        """
        assert self.rating_matrix is not None, 'Rating matrix is None'

        user_mean_ratings =self.rating_matrix.mean(axis  =1)
        assert len(user_mean_ratings) == self.rating_matrix.shape[0], 'Some thing went wrong'
        
        return {userid:user_mean_ratings[userid] for userid in  self.rating_matrix.index}

    def pearson_correlation(self, a: list, b: list ,mean_a:float = None, mean_b:float = None) -> float:
        """
        Compute the pearson correlation coefficient between a and b

        a: a list of rating of user/item a
        b: similar to a
        mean_a, mean_b: mean of a and b, if not provided, then it will be computed using a and b
        return the Pearson(a, b)
        """
        if a is None:
            mean_a = np.mean(a)
        if b is None:
            mean_b = np.mean(b)

        numerator = np.sum([(a_i - mean_a)*(b_i - mean_b) for a_i, b_i in zip(a, b)])
        denominator = np.sqrt(np.sum([(a_i - mean_a)**2 for a_i in a])) * np.sqrt(np.sum([(b_i - mean_b)**2 for b_i in b]))
        return numerator/denominator

    def cosine_similarity(self, a:list, b:list, mean_a = None, mean_b = None) -> float:
        """
        compute cosine similarity between a and b
        return: cosine(a, b)
        mean_a and mean_b is just for compatibility in other method.
        """
        return np.sum([ai*bi for ai, bi in zip(a, b)])/(np.sqrt(np.sum([ai**2 for ai in a])) * np.sqrt(np.sum([bi**2 for bi in b])))

    def get_rated_items(self, userid:int) -> list:
        """
        get all rated items by userid

        return a list of items (column name)
        """
        user_row = self.rating_matrix.loc[userid, :]
        # user_row is a series, can only be indexed by its column name
        items  = [x for x in self.rating_matrix.columns if user_row.loc[x] > 0]
        return items

class UserBasedCF(CollaborativeFiltering):

    def __init__(self, data:pd.DataFrame, k_neighbors:int = 10,
                 rating_matrix_row:str = None, rating_matrix_column:str = None, 
                 rating_matrix_value:str = None, movies_data:dict = None) -> None:
        """
        User-based Collaborative filtering algorithm
        params:
            data: pandas dataframe contains the data
            k_neighbors: number of most similar neighbors use to average the ratings over.
            rating_matrix_row/column/value: the name of the column in data use to create rating_matrix
            movie_data: a dict {movieid:movie_name} of all movies in the database

        """
        super().__init__(data, k_neighbors, rating_matrix_row, rating_matrix_column, 
                rating_matrix_value, movies_data)

    def get_mutually_rated_items(self, user1: int, user2: int) -> dict:
        """
        find the set of mutually observed rating between user1 and user2
        """
        user1_rated_items = self.get_rated_items(user1)
        user2_rated_items = self.get_rated_items(user2)
        mutually_rated_items = np.intersect1d(user1_rated_items, user2_rated_items, assume_unique=True )
        return mutually_rated_items

    def compute_similarity_score(self, target_user: int, similarity_metric:str) -> dict:
        """
        compute the similar score between target_user and all other users

        param:
            target_user: id of the target user
            metric: 'Pearson' or 'Cosine'

        return: a dict, its keys are user ids, its values are the metric scores
        """
        score_function = self.pearson_correlation if similarity_metric == 'Pearson' else self.cosine_similarity
        # k closest users to the user_id at hand, w.r.t one specific item
        scores = dict.fromkeys(self.rating_matrix.index, 0)
        items_rated_by_target_user = self.get_rated_items(target_user)
        for user in scores.keys():
            if target_user == user:
                scores[user] = 1
                continue
            items_rated_by_this_user = self.get_rated_items(user)
            mutually_rated_items = np.intersect1d(items_rated_by_target_user, items_rated_by_this_user)
            # if there is no common rated item between 2 user, the similar is 0
            if len(mutually_rated_items) == 0:
                continue
            else:
                
                scores[user] = score_function(self.rating_matrix.loc[target_user, mutually_rated_items],
                                             self.rating_matrix.loc[user, mutually_rated_items],
                                             self.user_mean_ratings[target_user], self.user_mean_ratings[user] )    

        return scores

    def predict_rating(
            self, target_user:int, target_item:int, k_neighbors:int = None,
            similarity_threshold: float = 0.5, mean_centered:bool= True, similarity_metric:str = 'Pearson') -> float:
        """
        Predict rating of target item of target user, using k most similar users
        to the target user that have rated the target item, and their similarity
        score > threshold. If not enough users satisfy the above conditions, then 
        less than k users will be used to predict the rating.
        
        param: 
            target_user: id of target user
            target_item: id of target item
            k_neighbors: number of neighbors use to predict rating
            threshold: the minumum score acceptable for a user to be a neighbor of target user
            mean_center: whether use the mean centered prediction fomular or not. If False, use 
                        the raw prediction formula.
            similarity_metric: either 'Pearson' or 'Cosine', the metric to compute similarity score

        return the predicted rating of the target item by the target user
        """
        

        if k_neighbors is None:
            k_neighbors = self.k_neighbors

        # check if the target user similarity score with other users has already computed, 
        # if not, compute and save to the self.all_similarity_score dictionary
        # otherwise, just retrieve it
        if self.all_similarity_score[target_user] is not None:
            if similarity_metric in self.all_similarity_score[target_user]:
                scores = self.all_similarity_score[target_user][similarity_metric]
            else:
                logger.info(f'Computing {similarity_metric} similarity of the target user and other users...')
                scores = self.compute_similarity_score(target_user = target_user, similarity_metric = similarity_metric)
                # save the score to memory
                self.all_similarity_score[target_user][similarity_metric] = scores
                logger.info('Done.')
        else:
            logger.info(f'Computing {similarity_metric} similarity of the target user and other users...')
            self.all_similarity_score[target_user] = {}
            scores = self.compute_similarity_score(target_user = target_user, similarity_metric = similarity_metric)
            self.all_similarity_score[target_user][similarity_metric] = scores
            logger.info('Done.')

        # sort the scores according to its values
        sorted_scores = {k: v for k, v in sorted(scores.items(), key=lambda item: item[1], reverse=True)}
        # filter out users that has similarity score < similarity_threshold
        sorted_scores = {user:score for user, score in sorted_scores.items() if score > similarity_threshold }

        neighbors = []
        # start finding neighbors
        for key in sorted_scores.keys():
            # if this user is not the target user and she rated the target item, append to neighbors
            if key != target_user and self.rating_matrix.loc[key, target_item] > 0:
                neighbors.append(key)
                # if there are enough neighbors, stop 
                if len(neighbors) >= k_neighbors:
                    break
        
        if len(neighbors) == 0:
            return 0
        # I'm not sure this is the right formula for the case there is only one neighbor!
        elif len(neighbors) == 1:
            return self.user_mean_ratings[target_user] + \
                 (sorted_scores[neighbors[0]]*(self.rating_matrix.loc[neighbors[0], target_item] - self.user_mean_ratings[neighbors[0]]))/np.abs(sorted_scores[neighbors[0]])

        if mean_centered:
            predicted = self.user_mean_ratings[target_user] + \
                np.sum([ sorted_scores[user]*(self.rating_matrix.loc[user, target_item] - self.user_mean_ratings[user]) for user in neighbors ] ) / \
                    np.sum([np.abs(sorted_scores[user]) for user in neighbors]) 

        else:
            # weighted average over raw ratings of neighbors
            predicted = np.sum([sorted_scores[user]*self.rating_matrix[user, target_item] for user in neighbors]) / \
                np.sum([sorted_scores[user] for user in neighbors])

        return predicted
                
    def recommend(self, target_user: int, num_items: int, similarity_metric: str = 'Pearson', 
                k_neighbors: int = None, similarity_threshold: float = 0.5,  mean_centered: bool = True, rating_threshold: int = 3) -> list:
        """
        recommend num_items to target_user

        param:
            target_user: the id of the target user
            num_items: number of items to recommend
            similarity_metric: either 'Pearson' or 'Cosine'
            k_neighbors: number of neighbors used to predict the rating
            similarity_threshold: the threshold of similarity metrics to choose neighbors
            mean_centered: if True, use the mean centered formula to predict the rating, otherwise use the raw formula
            rating_threshold: recomend items if its predicted rating > rating_threshold
        return a list of item id recommended by the algorithm

        """
        assert similarity_metric in ['Pearson', 'Cosine'], "similarity_metric must be 'Pearson' or 'Cosine'"

        predicted_rating = self.predict_ratings(target_user, similarity_metric, k_neighbors, similarity_threshold, mean_centered)

        logger.info('Predict rating done. Recommending promising items')

        if len(predicted_rating) > 1:
            predicted_rating = {k: v for k, v in sorted(predicted_rating.items(), key=lambda item: item[1], reverse=True)}
        
        # filter out the items that have predicted rating < rating_threshold
        recommending_items = {item:predicted_rating[item] for item in predicted_rating.keys() if predicted_rating[item] > rating_threshold}
        if len(recommending_items) > num_items:
           recommending_items = {item:predicted_rating[item] for item in list(recommending_items.keys())[:num_items] }
           
        logger.info(f'These are {num_items} promising items for the target user {target_user} ')
        if self.movies_data is not None:
            return {self.movies_data[item]:score for item, score in recommending_items.items()}
        else:
            return recommending_items

    def predict_ratings(self, target_user:int, similarity_metric:str = 'Pearson', 
                k_neighbors:int = None, similarity_threshold:float = 0.5,  mean_centered:bool = True, ) -> dict:
        """
        Predict ratings of the target user for all the items she did not rated
        return: a dict {item:predicted_rating} for all item
        """
        if k_neighbors is None:
            k_neighbors = self.k_neighbors
        rated_items = self.get_rated_items(target_user)
        not_rated_items = list(set(self.rating_matrix.columns) - set(rated_items))
        # consider the case the user has rated all the items, but I doubt this if will ever entered
        if len(not_rated_items) == 0:
            print('There is nothing left for this user')
            return []

        logger.info('Start predict rating...')
        rating_predicted = {}
        for item in not_rated_items:
            rating_predicted[item] = self.predict_rating(target_user = target_user, target_item = item, k_neighbors = k_neighbors,
                                                        similarity_threshold = similarity_threshold, mean_centered = mean_centered,
                                                        similarity_metric=similarity_metric)

        return rating_predicted

class ItemBasedCF(CollaborativeFiltering):
    def __init__(self, data:pd.DataFrame, k_neighbors:int= 10, rating_matrix_row:str=None,
                    rating_matrix_column:str = None, rating_matrix_value:str = None, movies_data:pd.DataFrame = None ) -> None:
        """
        Item-based collaborative filtering algorithm.
        Similarity metric to compare 2 items can be either Pearson or Adjusted Cosine
        The rating matrix in this class is still (m,n) of m users and n items

        params:
            data: pandas dataframe contains the data
            movie_data: a dict {movieid:movie_name} of all movies in the database
            k_neighbors: number of most similar neighbors use to average the ratings over.
            rating_matrix_row/column/value: the name of the column in data use to create rating_matrix
        """
        # the super class __init__ method create the rating matrix, compute the user mean rating.
        logger.info('Creating rating matrix')
        super( ).__init__(data, k_neighbors, rating_matrix_row, rating_matrix_column, 
                rating_matrix_value, movies_data)

        # create the user-mean centered rating_matrix version of the raw rating_matrix
        logger.info('creating centered version of the rating matrix')
        self.centered_rating_matrix = self.get_centered_rating_matrix()

    def get_centered_rating_matrix(self) -> pd.DataFrame:
        """
        Centering the rating matrix by its row mean
        """
        assert 'user_mean_ratings' in dir(self), 'Not found the user_mean_ratings dict'  
        centered_rating_matrix = copy.deepcopy(self.rating_matrix)
        for user in self.user_mean_ratings:
            centered_rating_matrix.loc[user, :] -= self.user_mean_ratings[user]
        return centered_rating_matrix

    def adjusted_cosine(self,a:list, b:list, )->float:
        """
        Compute the ajusted Cosine between a and b
        a and b are expected to be already centered by the user mean rating
        """
        return np.sum([(ai*bi) for ai, bi in zip(a, b)])/(np.sqrt(np.sum([ai**2 for ai in a])) * np.sqrt(np.sum([bi**2 for bi in b])))

    def get_user_rated_item(self, item:int) -> list:
        """
        Get a list of users who rated the item
        """
        item_col = self.rating_matrix.loc[:, item]
        # user_row is a series, can only be indexed by its column name
        users  = [x for x in self.rating_matrix.index if item_col.loc[x] > 0]
        return users

    def compute_item_mean_ratings(self, ) -> dict:
        """
        Compute the average rating for each item
        """
        self.rating_matrix = self.rating_matrix.replace(0, np.nan)
        means = self.rating_matrix.mean(axis = 0)
        assert len(means) == self.rating_matrix.shape[1], 'Some thing went wrong'
        item_mean_ratings = {item: means[item ] for item in self.rating_matrix.columns}
        return item_mean_ratings

    def compute_similarity_score(self, target_item:int, similarity_metric:str = 'AdjustedCosine' ) -> dict:
        """
        compute the similar score between target_item and all other items

        param:
            target_item: id of the target item
            metric: 'Pearson' or 'AdjustedCosine'

        return: a dict, its keys are user ids, its values are the metric scores
        """
        # score_function = self.adjusted_cosine  if similarity_metric == 'AdjustedCosine' else self.pearson_correlation
        scores = dict.fromkeys(self.rating_matrix.columns, 0)

        # get all users who rated the target item
        users_rated_target_item = self.get_user_rated_item(target_item)
        #loop over all the items to compute the similarity score with the target item
        for item in scores.keys():
            if target_item == item:
                scores[item] = 1
                continue
            users_rated_this_item = self.get_user_rated_item(item)
            mutually_rated_users = np.intersect1d(users_rated_target_item, users_rated_this_item, assume_unique= True)
            # if there is no common users who both rated target item and the current item, set the score = 0
            if len(mutually_rated_users) == 0:
                continue
            else:
                if similarity_metric == 'AdjustedCosine':
                    scores[item] = self.adjusted_cosine(self.centered_rating_matrix.loc[mutually_rated_users, target_item],
                                             self.centered_rating_matrix.loc[mutually_rated_users, item] ) 
                elif similarity_metric == 'Pearson':
                    scores[item] = self.pearson_correlation(self.rating_matrix.loc[mutually_rated_users, target_item], 
                                                            self.rating_matrix.loc[mutually_rated_users, item], 
                                                            self.item_mean_ratings[target_item], self.item_mean_ratings[item])
        return scores

    def predict_rating(
            self, target_user:int, target_item:int, k_neighbors:int = None,
            similarity_threshold:float = 0.5,  similarity_metric:str = 'AdjustedCosine') -> float:
        """
        Predict rating of the target user to the target item.

        param:
            target_user: id of target user
            target_item: id of target item
            k_neighbors: number of neighbor use to predict rating
            similarity_threshold: only consider an item as a neighbor if its similarity score with the target item > this threshold
            similarity_metric: either 'AdjustedCosine' or 'Pearson'. 

        Return the predicted rating of the target user to the target item
        Either the similarity metric is Ajusted Cosine or Pearson,
        the prediction formular will be the raw weighted average, not the 
        mean centered prediction formula as in the UserBasedCF case.
        """
        if k_neighbors == None:
            k_neighbors = self.k_neighbors

        # retrieve or compute (if have to) similarity scores between target item and all other items
        if self.all_similarity_score[target_item] is not None:
            if similarity_metric in self.all_similarity_score[target_item].keys():
                
                scores = self.all_similarity_score[target_item][similarity_metric]
                # logger.info(f'Retrieving similarity score {len(scores)}')
            else:
                logger.info(f'Computing {similarity_metric} similarity of the target item and other items...')
                scores = self.compute_similarity_score(target_item = target_item, similarity_metric = similarity_metric)
                # save the score to memory
                self.all_similarity_score[target_item][similarity_metric] = scores
                logger.info('Done.')
        else:
            logger.info(f'Computing {similarity_metric} similarity of the target item and other items...')
            self.all_similarity_score[target_item] = {}
            scores = self.compute_similarity_score(target_item = target_item, similarity_metric = similarity_metric)
            self.all_similarity_score[target_item][similarity_metric] = scores
            logger.info('Done.')

        # start finding neighbors, an item is only accepted as a neighbor of the target item
        # for the target user if that user has rated that item, and the similartity score is higher
        # than the similarity_threshold
        # a smart way is only loop over the rated items of the target user


        rated_items_by_target_user = self.get_rated_items(target_user)
        # neighbor items are items that have rated by the target user AND have similarity score > similarity threshold
        neighbors = {item:scores[item] for item in rated_items_by_target_user if scores[item] > similarity_threshold }

        predicted = -1
        if len(neighbors) == 0:
            return predicted
        elif len(neighbors) == 1:
            return self.rating_matrix.loc[target_user, target_item]
        elif len(neighbors) < k_neighbors:
            # average over raw rating, no matter it is AdjustedCosine or Pearson
            predicted = np.sum([score*self.rating_matrix.loc[target_user, item] for item, score in neighbors.items() ])/np.sum(list(neighbors.values()))
        else:
            # there are more acceptable neighbor items than we need, so we will choose the k_neighbors with the largest similarity score.
            sorted_neighbors = {k: v for k, v in sorted(neighbors.items(), key=lambda item: item[1], reverse=True)}
            neighbors_items = list(sorted_neighbors.keys())[:k_neighbors]
            neighbors = {item:sorted_neighbors[item] for item in neighbors_items }
            predicted = np.sum([score*self.rating_matrix.loc[target_user, item] for item, score in neighbors.items()])/np.sum(list(neighbors.values()))
        return predicted


    def predict_ratings(self, target_item:int, similarity_metric:str = 'AdjustedCosine',
                  k_neighbors:int = 10, similarity_threshold:float = 0.5, ) -> dict:
        """
        Predict ratings of all the users who did not rate the target item for the target item.
        return a dict {user:predicted_rating}
        """
        if k_neighbors is None:
            k_neighbors = self.k_neighbors

        assert similarity_metric in ['AdjustedCosine', 'Pearson'], "similarity_metric can only be 'AdjustedCosine' or 'Pearson'"
        # compute item mean rating if the similarity metric is Pearson
        if similarity_metric == 'Pearson':
            self.item_mean_ratings = self.compute_item_mean_ratings()            
        # We only consider the users who did not rate the target item
        users_not_rated_target_item = list(set(self.rating_matrix.index) - set(self.get_user_rated_item(target_item)) )
        # for each user that did not rate the target_item, predict the rating of that user to the target item
        logger.info('Start predict rating...')
        predicted_rating = dict.fromkeys(users_not_rated_target_item, 0)

        for user in users_not_rated_target_item:
            predicted_rating[user] = self.predict_rating(user, target_item, k_neighbors, similarity_threshold, similarity_metric )
        logger.info('Predict rating done. Recommending promising users')

        # sort the predicted rating
        predicted_rating =  {k: v for k, v in sorted(predicted_rating.items(), key=lambda item: item[1], reverse=True)}
        # filter out the predicted rating that < rating_threshold

        return predicted_rating

    def recommend(self, target_item:int, num_users:int, similarity_metric:str = 'AdjustedCosine',
                  k_neighbors:int = 10, similarity_threshold:float = 0.5, rating_threshold:int = 4 ) -> dict:
        """
        Find k most promising users for the target item

        param:
            target_user: id of target user
            target_item: id of target item
            k_neighbors: number of neighbor use to predict rating
            similarity_threshold: only consider an item as a neighbor if its similarity score with the target item > this threshold
            similarity_metric: either 'AdjustedCosine' or 'Pearson'. 
            rating_threshold: only recommend a user if her predicted rating for the target item > this threshold

        return a dict of recommended items and its predicted rating

        The steps for Item-Based CF: 
            1. Given a target item, compute the similarity score between the target item
                and all other items.
            2. For each user in the database that does not rate the target item:
                    Find k rated item by that user, such that they are most similar to the target item.
                    Predict the rating of this user to the target item, using the weighted average formula
                    Save the predicted rating to a dict {userid: predicted_rating}
            3. Sort the previous saved dict, then take out k users that have the highest predicted rating. 
                Those users are then considered the most promising users. 
        """

        predicted_rating = self.predict_ratings(target_item, similarity_metric, k_neighbors, similarity_threshold)
        
        # filter out the predicted rating that < rating_threshold
        predicted_rating = {user:rating for user, rating in predicted_rating.items() if rating > rating_threshold}

        if len(predicted_rating) > num_users:
            predicted_rating = {user:predicted_rating[user] for user in list(predicted_rating.keys())[:num_users] }

        logger.info(f'These are {num_users} promising users for the target item {target_item}')
        if self.movies_data is None:
            return predicted_rating
        else:
            return {user:predicted_rating[user] for user in list(predicted_rating.keys())[:num_users] }

if __name__ == '__main__':
    # these are my test data
    data_dict = {'userID': [1,1,1,1,1, 2,2,2,2,2, 3,3,3,3,3, 4,4,4,4,4], 
             'movieID': [1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5, 1,2,3,4,5], 
             'rating': [np.nan ,4,1,1, np.nan,  1,2, 4,np.nan, 1,  5, 5, 3,4,np.nan, 5,5,1, np.nan, 1]}
    data = pd.DataFrame.from_dict(data_dict)
    recommender = ItemBasedCF(data, 2, 'userID', 'movieID', 'rating')
    # print(recommender.rating_matrix)
    print(recommender.recommend(4, 1, 'Pearson', 1))