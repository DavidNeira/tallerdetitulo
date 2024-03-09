import pandas as pd
import os

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
df_books= pd.read_csv(original_path + "/books_data.csv", sep=",")
df_books = df_books[['Title', 'categories']]

df_ratings= pd.read_csv(original_path + "/books_rating.csv", sep=",")
df_ratings = df_ratings[['Title', 'User_id', 'review/score']]  # Corregir el uso de paréntesis aquí
df_ratings_filtered = df_ratings[df_ratings['review/score'] >= 4.0]

df_resultado = pd.merge(df_books, df_ratings_filtered, on="Title")
df_resultado = df_resultado.fillna('-')
cantidad_registros = df_resultado.shape[0]
print("Cantidad de registros:", cantidad_registros)
# CANTIDAD DE REGISTROS 2.392.959
df_resultado.to_csv('books_amazon.csv', index=False)

#df_muestra = df_resultado.sample(n=1000000, random_state=42)  # random_state para reproducibilidad
#cantidad_registros = df_muestra.shape[0]
print("Cantidad de registros:", cantidad_registros)

# df_muestra.to_csv('books_amazon1000k.csv', index=False)


print("Head\n", df_resultado.head())
print("FIN")

####################
# K-core filtering #
####################