import argparse
import json
import numpy as np

# Build an argument parser to process input arguments. It will accept two users and the type of score that it needs to compute similarity
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Compute similarity score')
    parser.add_argument('--user1', dest='user1', required=True, help='First user')
    parser.add_argument('--user2', dest='user2', required=True, help='Second user')
    parser.add_argument("--score-type", dest="score_type", required=True,
                        choices=['Euclidean', 'Pearson'], help='Similarity metric to be used')
    return parser

# compute the Euclidean distance score between user1 and user2
def euclidean_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')
        
    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')
        
    # movies rated by both user1 and user2
    common_movies = {}
    
    # extract movies rated by both users
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1
    
    # if there are no common movies between users, then similarity = 0
    if len(common_movies) == 0:
        return 0
    
    # compute the squared differences between the ratings and use it to compute Euclidean score
    squared_diff = []
    
    for item in dataset[user1]:
        if item in dataset[user2]:
            squared_diff.append(np.square(dataset[user1][item] - dataset[user2][item]))
    return 1 / (1 + np.sqrt(np.sum(squared_diff))) 

# compute Pearson correlation score between user1 and user2
def pearson_score(dataset, user1, user2):
    if user1 not in dataset:
        raise TypeError('Cannot find ' + user1 + ' in the dataset')
    if user2 not in dataset:
        raise TypeError('Cannot find ' + user2 + ' in the dataset')
        
    # movies rated by both user1 and user2
    common_movies = {}
    # extract movies rated by both users
    for item in dataset[user1]:
        if item in dataset[user2]:
            common_movies[item] = 1
    
    num_ratings = len(common_movies)
    
    # if there are no movies in common, then score = 0
    if num_ratings == 0:
        return 0
    
    # calculate the sum of ratings of all common movies 
    user1_sum = np.sum([dataset[user1][item] for item in common_movies])
    user2_sum = np.sum([dataset[user2][item] for item in common_movies])
    
    # calculate the sum of squares of the ratings of all common movies
    user1_squared_sum = np.sum([np.square(dataset[user1][item]) for item in common_movies])
    user2_squared_sum = np.sum([np.square(dataset[user2][item]) for item in common_movies])
    
    # calculate the sum of products of the ratings of common movies
    sum_of_products = np.sum([dataset[user1][item] * dataset[user2][item] for item in common_movies])
    
    # calculate the Pearson score
    Sxy = sum_of_products - (user1_sum * user2_sum / num_ratings)
    Sxx = user1_squared_sum - np.square(user1_sum) / num_ratings
    Syy = user2_squared_sum - np.square(user2_sum) / num_ratings
    
    # if there is no deviation then the score is 0
    if Sxx * Syy == 0:
        return 0
    
    # return Pearson score
    return Sxy / np.sqrt(Sxx * Syy)

# define the main function and parse the input argument
if __name__=='__main__':
    args = build_arg_parser().parse_args()
    user1 = args.user1
    user2 = args.user2
    score_type = args.score_type
    
    # load ratings from 'ratings.json' into a dictionary
    ratings_file = 'ratings.json'
    
    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())
        
    # compute similarity score based on the input arguments
    if score_type == 'Euclidean':
        print("\nEuclidean score:")
        print(euclidean_score(data, user1, user2))
    else:
        print("\nPearson score:")
        print(pearson_score(data, user1, user2))
        
    

    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    