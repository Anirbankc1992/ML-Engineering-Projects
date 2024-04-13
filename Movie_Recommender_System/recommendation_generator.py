from re import sub
import argparse
from Recommenders.recommender import MovieAnalyzer, TitleRecommender, SimilarityUserRecommender
import yaml
import warnings; warnings.simplefilter('ignore')


source_file = 'recommend_config.yaml'

with open(source_file, "r") as file:
    src = yaml.safe_load(file)


mv = MovieAnalyzer(**src['dataset'])
average_ratings,vote_counts,ratings_with_count = mv()
df = mv.df

parser = argparse.ArgumentParser(description="Recommendation System Argument Parser",
                                epilog="Type of recommendation: title or user_similarity",
                                formatter_class=argparse.ArgumentDefaultsHelpFormatter)
subparsers = parser.add_subparsers(
    dest="command",
    title="Title or User Similarity Recommendation",
    help="choose between either Title or User Similarity Recommendation"
)

title = subparsers.add_parser(
    "title", help="Title recommendation by calculating cosine similarities on tfidf matrix"
)
    
title.add_argument(
    "-title",
    "-t",
    type=str,
    help="Specify title name whose similar recommended movies you need!",
)

similar_users = subparsers.add_parser(
    "similar_users", help="Find users with similar user-item interactions"
)
similar_users.add_argument(
    "-user_id",
    "-u",
    type=int,
    help="Specify user_id for whom you want to recommend movie for!",
)
    
args = parser.parse_args()

if args.command == "title":
    
    # Perform title recommendation logic
    print(f"Title recommendation requested for '{args.title}'")
    
    recommender = TitleRecommender(df,src['title_recommender']['records'])
    title_records = dict(title=args.title,k= src['title_recommender']['k'])
    
    recommended_titles = recommender.recommend_by_title(**title_records)
    
    normalised_title = sub(r'[^\w\s]', '', args.title).lower().replace(' ', '_')
    recommended_titles.to_csv(src['output_data_folder']['title'].format(normalised_title),index=False)
  
    
elif args.command == "similar_users":
    
    # Perform user similarity recommendation logic
    print(f"User similarity recommendation requested for user ID {args.user_id}")
    
    similar_users = (dict(user_id=args.user_id) | src['user_recommender'])
    similarity_calculator = SimilarityUserRecommender(df)
    
    similarities = similarity_calculator.get_similar_users(**similar_users)
    similarities.to_csv(src['output_data_folder']['user'].format(args.user_id),index=False)

        








