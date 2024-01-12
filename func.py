import numpy as np
import pandas as pd
import functions as func

def song_recommender(title, artist):
    
    #Import
    full_df = pd.read_csv("full_df.csv")
    full_df['track_link'] = ["https://open.spotify.com/track/" + str(i) for i in full_df['id'].values]
    full_df.drop(columns = 'Unnamed: 0', inplace = True)
    
    cluster_df = pd.read_csv("X_umap_transformed_df_UMAP_HDBSCAN.csv")
    cluster_df.drop(columns = 'Unnamed: 0', inplace = True)
    
    final_df = pd.concat([full_df,cluster_df],axis=1)
    final_df_num = final_df.select_dtypes(np.number)
    
    #searching for song id
    id_input_song = func.search_song(title, artist, limit=1)
    
    #searching for audio features
    id_input_song_audio_features = pd.DataFrame(func.get_single_audio_features(id_input_song))
    id_input_song_audio_features_num = id_input_song_audio_features.select_dtypes(np.number)
    
    #apply scaler and umap
    input_scaled_umap = func.apply_scale_umap(id_input_song_audio_features_num)
    input_scaled_umap_df = pd.DataFrame(input_scaled_umap[1])
    
    #import umap_df and concat to full_df
    umap_df = pd.read_csv("X_umap_transformed_df_UMAP_HDBSCAN.csv")
    umap_df_new = umap_df.drop(columns = ['Unnamed: 0','cluster'])
    full_df = pd.concat([full_df,umap_df],axis = 1)
    full_df.drop(columns = 'Unnamed: 0', inplace = True)
   
    #computing distance_matrix and closest distance 
    from scipy.spatial import distance_matrix
    distance = distance_matrix(input_scaled_umap_df, umap_df_new, p=2, threshold=1000000)
    closest_distance_index = np.argmin(distance)
    
    #define cluster of the imported song
    input_cluster = cluster_df.iloc[closest_distance_index]
    input_cluster = input_cluster[2]

    
    #checking if the song is in the Hot Song list and print 5 samples
    hot_df = full_df[(full_df['dataset']=='hot')]

    if id_input_song in hot_df['id']:
        sample_hot = full_df[(full_df['dataset']=='hot')&(full_df['cluster']==input_cluster)].sample(5)
        output = pd.DataFrame(sample_hot[['title','artist','track_link']].reset_index(drop=True))
    else:
        sample_not_hot = full_df[(full_df['dataset']=='not_hot')&(full_df['cluster']==input_cluster)].sample(5)
        output = pd.DataFrame(sample_not_hot[['title','artist','track_link']].reset_index(drop=True))
    return output.to_html(index=False)  