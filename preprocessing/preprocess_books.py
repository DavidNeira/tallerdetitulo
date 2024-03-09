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

dir_path = "../data/books/"
original_path = dir_path + "original_data"
final_path = dir_path
rating_df = pd.read_csv(original_path + "/Ratings.csv", sep=";")
df = pd.concat([rating_df], ignore_index=True)
df = df[(df["Rating"] >= 8) | (df["Rating"] == -1)] # Remove interactions corresponding to low (< 4) ratings

print("Head\n", df.head())
####################
# K-core filtering #
####################


filtered_user_ids = set(df['User-ID']) # Update list of users filtered to remove those with no interaction
filtered_item_ids = set(df['ISBN']) # Update list of items filtered to remove those with no interaction
print("Number of users:", len(filtered_user_ids))
print("Number of items:", len(filtered_item_ids))


training_tuples = []
validation_tuples = []
test_tuples = []
user_list = set(df['User-ID'])
item_list = set(df['ISBN'])
user_dict = {u: user_id for (user_id, u) in enumerate(user_list)}
item_dict = {p: item_id for (item_id, p) in enumerate(item_list)}

for (u, user_id) in user_dict.items():
    user_query_df = df[df['User-ID'] == u].sort_values(by=['Rating'])
    if user_id % 1000 == 0:
        print("Number of users processed: " + str(user_id), flush=True)
    n_interaction = user_query_df.shape[0]
    n_test = int(test_ratio * n_interaction)
    n_validation = int(validation_ratio * n_interaction)
    n_training = n_interaction - n_validation - n_test

    for interaction_count, (row, interaction) in enumerate(user_query_df.iterrows()):
        item_id = item_dict[interaction['ISBN']] # Item interacted
        rating = interaction['Rating'] # Rating of the interaction or -1 if no rating
        rating = "-" if rating == -1 else str(rating)

        # Process the query and add it to training, validation or test set
        if interaction_count < n_training: # Training set
            training_tuples.append((user_id, item_id, rating))
        elif interaction_count >= n_training and interaction_count < n_training + n_validation: # Validation set
            validation_tuples.append((user_id, item_id, rating, ))
        else: # Test set
            test_tuples.append((user_id, item_id, rating))

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
    f.write("user_id\titem_id\tquery\n")
    for (user_id, item_id, rating) in training_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + rating +"\n")


# Save validation file
training_path = final_path + "/valid.txt"
with open(training_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\tquery\n")
    for (user_id, item_id, rating) in validation_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + rating +"\n")


# Save test file
training_path = final_path + "/test.txt"
with open(training_path, "w+", encoding='utf-8') as f:
    f.write("user_id\titem_id\tquery\n")
    for (user_id, item_id, rating) in test_tuples:
        f.write(str(user_id) + "\t" + str(item_id) + "\t" + rating +"\n")
