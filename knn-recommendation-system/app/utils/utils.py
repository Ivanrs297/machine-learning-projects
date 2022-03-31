from thefuzz import fuzz

# Function to match the input movie name to database movie names
def get_movie_name_match(movie_name_to_id, fav_movie):

    matched_tuples = []

    for title, id in movie_name_to_id.items():

        # get fuzzy ratio from movies
        ratio = fuzz.ratio(title.lower(), fav_movie.lower())

        if ratio >= 35:
            matched_tuples.append((title, id, ratio))
            
    # sort the matched tuples
    # key is the ratio at position 2
    matched_tuples = sorted(matched_tuples, key = lambda x: x[2])

    # Descending order
    matched_tuples = matched_tuples[::-1]
    
    top_match_tuple = None

    if not matched_tuples:
        print("No match found")
    else:
        print(f"Found matches {matched_tuples}")

        # return top match
        top_match_tuple =  matched_tuples[0][1]
        
    return top_match_tuple


