import torch

# Cargar el modelo desde el estado guardado
saved_model = torch.load('lastfm-novalid-LightGCN-l2-ed0.1-e100-b1024-r0.001-h64-n1-w0.0-s2019.t7')

# Extraer el ranking directamente del estado del modelo
ranking = saved_model['model']['user_embeddings.weight']
item_ranking = saved_model['model']['item_embeddings.weight']
# Imprimir solo los 10 primeros elementos del ranking para los primeros 5 usuarios
num_users_to_print = 5
for i in range(num_users_to_print):
    user_weights = ranking[i]
    print(f"\nRanking para el Usuario {i + 1}:")
    user_ranking = user_weights.argsort(descending=True)
    for j, item_id in enumerate(user_ranking):
        print(f"Posición {j + 1}: Ítem {item_id.item()} con peso {user_weights[item_id].item()}")


num_items_to_print = 5
for j in range(num_items_to_print):
    item_weights = item_ranking[j]
    print(f"\nRanking para el Ítem {j + 1}:")
    item_ranking_indices = item_weights.argsort(descending=True)
    for i, user_id in enumerate(item_ranking_indices):
        print(f"Posición {i + 1}: Usuario {user_id.item()} con peso {item_weights[user_id].item()}")