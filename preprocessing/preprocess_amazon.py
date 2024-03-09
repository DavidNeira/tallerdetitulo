import pandas as pd

# Parameters
## Filter
min_i = 10
min_u = 10
## Split
training_ratio = 0.6
validation_ratio = 0.2
test_ratio = 0.2

#################
# Load the data #
#################

print("Loading data...", flush=True)

dir_path = "../data/amazon_books/"
original_path = dir_path + "original_data"
final_path = dir_path
#read the books data
df =  pd.read_csv(original_path + "/books_amazon1000K.csv", sep=",")
df['categories'] = df['categories'].apply(lambda x: str(x).strip("[]"))
df = df[~df['categories'].isnull()] # Remove NaN tags
df = df.fillna('-')


print("-------------------------------")
df = df[['Title', 'User_id', 'review/score','categories']] 
print("Head\n", df)
print("-------------------------------\n")


####################
# K-core filtering #
####################

category_user_ids = set(df[df['categories'] != -1].User_id)
df = df[df['User_id'].isin(category_user_ids)]

print("Head\n", df.head())

# Filter out items with less than min_i interactions
print("Filtering out items...", flush=True)
dist_df = df['Title'].value_counts()

dist_df = dist_df[dist_df >= min_i]
filtered_item_ids = dist_df.keys()
df = df[df['Title'].isin(filtered_item_ids)]

# Filter out users with less than min_u interactions
print("Filtering out users...", flush=True)
dist_df = df['User_id'].value_counts()
dist_df = dist_df[dist_df >= min_u]
filtered_user_ids = dist_df.keys()
df = df[df['User_id'].isin(filtered_user_ids)]


filtered_user_ids = set(df['User_id']) # Update list of users filtered to remove those with no interaction
filtered_item_ids = set(df['Title']) # Update list of items filtered to remove those with no interaction
print("Number of users:", len(filtered_user_ids))
print("Number of items:", len(filtered_item_ids))


training_tuples = []
validation_tuples = []
test_tuples = []
user_list = set(df['User_id'])
item_list = set(df['Title'])
user_dict = {u: user_id for (user_id, u) in enumerate(user_list)}
item_dict = {p: item_id for (item_id, p) in enumerate(item_list)}

for (u, user_id) in user_dict.items():
    user_query_df = df[df['User_id'] == u].sort_values(by=['review/score'])
    if user_id % 1000 == 0:
        print("Number of users processed: " + str(user_id), flush=True)
    n_interaction = user_query_df.shape[0]
    n_test = int(test_ratio * n_interaction)
    n_validation = int(validation_ratio * n_interaction)
    n_training = n_interaction - n_validation - n_test

    for interaction_count, (row, interaction) in enumerate(user_query_df.iterrows()):
        item_id = item_dict[interaction['Title']] # Item interacted
        rating = interaction['review/score'] # Rating of the interaction or -1 if no rating
        rating = "-" if rating == -1 else str(rating)

        #added tag interaction by 
        tag = interaction['categories'] # ID of the tag or -1 if no tag
        tag_text = "-" if tag == -1 else str(tag)
        tag_text = tag_text.replace('[', '"').replace(']', '"')
        # tag_text = '"' + tag_text.replace('\t', ' ') + '"'

        # Process the query and add it to training, validation or test set
        if interaction_count < n_training: # Training set
            training_tuples.append((user_id, item_id, rating, tag_text))
        elif interaction_count >= n_training and interaction_count < n_training + n_validation: # Validation set
            validation_tuples.append((user_id, item_id, rating,tag_text))
        else: # Test set
            test_tuples.append((user_id, item_id, rating,tag_text))

n_user = len(user_list)
n_item = len(item_dict)

print("Training", len(training_tuples), flush=True)
print("Validation", len(validation_tuples), flush=True)
print("Test", len(test_tuples), flush=True)


##########################
# Save preprocessed data #
##########################

print("Saving preprocessed data...", flush=True)

# Save data size file
data_size_path = final_path + "/data_size.txt"
with open(data_size_path, "w+", encoding='utf-8') as f:
    f.write(str(n_user) + "\t" + str(n_item) + "\n")

# Save training file
training_path = final_path + "/train.txt"
with open(training_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\trating\tquery\n")
    for (user_id, item_id, rating, tag_text) in training_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + rating +"\t"+str(tag_text)+"\n")


# Save validation file
training_path = final_path + "/valid.txt"
with open(training_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\trating\tquery\n")
    for (user_id, item_id, rating, tag_text) in validation_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + rating +"\t"+str(tag_text)+"\n")


# Save test file
training_path = final_path + "/test.txt"
with open(training_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\trating\tquery\n")
    for (user_id, item_id, rating, tag_text) in test_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + rating +"\t"+str(tag_text)+"\n")
