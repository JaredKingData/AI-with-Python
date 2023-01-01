import argparse
import json
import numpy as np

from compute_scores import pearson_score

# define a function to parse the input arguments. The only input argument would be the name of the user
def build_arg_parser():
    parser = argparse.ArgumentParser(description='Find users who are similar to input user ')
    parser.add_argument('--user', dest='user', required=True, help='Input user')
    return parser

# finds users in the dataset that are similar to the input user
def find_similar_users(dataset, user, num_users):
    if user not in dataset:
        raise TypeError('Cannot find ' + user + ' in the dataset')
        
    # compute the Pearson score between input user and all the users in the dataset
    scores = np.array([[x, pearson_score(dataset, user, x)] for x in dataset if x != user])
    
    # sort the scores in descending order
    scores_sorted = np.argsort(scores[:, 1])[::-1]
    
    # extract the top 'num_users' scores
    top_users = scores_sorted[:num_users]
    return scores[top_users]

# define the main function and parse the input arguments to extract the name of the user
if __name__ == '__main__':
    args = build_arg_parser().parse_args()
    user = args.user
    
    # load the data from ratings.json containing the names of people and their ratings for various movies
    ratings_file = 'ratings.json'
    
    with open(ratings_file, 'r') as f:
        data = json.loads(f.read())
        
        # find top 3 users who are similar to the user specified by the input argument (You can change it to any number of users of your choice), and print the output along with the scores.
        print('\nUsers similar to ' + user + ':\n')
        similar_users = find_similar_users(data, user, 3)
        print('User\t\t\tSimilarity score')
        print('-'*41)
        for item in similar_users:
            print(item[0], '\t\t', round(float(item[1]), 2))
            
# EXAMPLE (RUN IN TERMINAL): !python "FindingSimilarUsersWithCollaborativeFiltering.py" --user "Clarissa Jackson"