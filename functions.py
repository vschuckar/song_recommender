import pandas as pd
import numpy as np
import sys
from config import *
import spotipy
import json
from spotipy.oauth2 import SpotifyClientCredentials
from time import sleep
import pickle



def search_song(title, artist, limit=1):
    search_query = f"track:{title} artist:{artist}"
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=Client_ID, client_secret=Client_Secret))
    song_id = sp.search(q=search_query, limit=limit)['tracks']['items'][0]['id']
          
    return song_id

def add_id(df):
    chunks = 50
    list_of_ids = []

    for i in range(0, len(df), chunks):
        chunk = df.iloc[i:i+chunks]
        print("Collecting IDs for chunk",int(i/chunks))

        for index, row in chunk.iterrows():
            title = row["title"]
            artist = row["artist"]
            try:
                id = search_song(title, artist,1)
                list_of_ids.append(id)
            except:
                print("Song not found!")
                list_of_ids.append("")
        sleep(30)
        print('sleep')
    df["id"] = list_of_ids
    return df

def get_audio_features(list_of_ids):
    chunk_size = 50
    feature_df = pd.DataFrame()
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=Client_ID, client_secret=Client_Secret))
    
    for i in range(0, len(list_of_ids), chunk_size):
        chunk = list_of_ids[i:i+chunk_size]
        print("Collecting audio features for chunk",int(i/chunk_size))
        
        try:
            my_dict = sp.audio_features(list_of_ids[i:i+chunk_size])
            df = pd.DataFrame(my_dict)
            feature_df = pd.concat([feature_df,df], ignore_index=True)
        except:
            print("Error retrieving features")
                
        print("Sleeping")
        sleep(30)
    
    
    return feature_df

def get_single_audio_features(id):
    
    sp = spotipy.Spotify(auth_manager=SpotifyClientCredentials(client_id=Client_ID, client_secret=Client_Secret))
    
    my_dict = sp.audio_features(id)
        
        
    return my_dict


def add_audio_features(df:pd.DataFrame, audio_features_df:pd.DataFrame):

    full_df = df.merge(audio_features_df, how="left", on='id')

    return full_df

def drop_empty_rows(df:pd.DataFrame, col:str):
    
    list_of_row_index = df[df[col] == ""].index.tolist()
    df = df.drop(list_of_row_index,axis=0)
    return df

def add_songs_features_run_all(df):   
    df = add_id(df)
    df = drop_empty_rows(df,'id')
    list_of_ids = df["id"].tolist()
    features = get_audio_features(list_of_ids)
    df = add_audio_features(df, features)
    return df

def apply_scale_umap(df:pd.DataFrame):
    
    #Scaling
    with open('pickle/scaler.pkl', "rb") as file:
        loaded_scaler = pickle.load(file)
        scale_df = loaded_scaler.transform(df)
    #UMAP
    with open('pickle/umap.pkl', "rb") as file:
        loaded_umap = pickle.load(file)
        umap_df = loaded_umap.transform(df)
        
 
    
    return [scale_df, umap_df]

def song_recommender():
    
    #Import
    full_df = pd.read_csv("full_df.csv")
    full_df.drop(columns = 'Unnamed: 0', inplace = True)
    cluster_df = pd.read_csv("X_umap_transformed_df_UMAP_HDBSCAN.csv")
    cluster_df.drop(columns = 'Unnamed: 0', inplace = True)
    
    final_df = pd.concat([full_df,cluster_df],axis=1)
    final_df_num = final_df.select_dtypes(np.number)
        
    #Enter title and artistname
    #title = input("Give me the title of a song:")
    #title = title.lower()
    loop = 0
    while loop == 0:
        title = input("Give me the title of a song:")
        title = title.lower()
        if title in full_df['title'].values:
            loop = 1
        else:
            print("Unfortunately your title was not found. Please try another one.")
            loop = 0
    loop = 0
    while loop == 0:
        artist = input("Tell me the name of the artist:")
        artist = artist.lower()
        if artist in full_df['artist'].values:
            loop = 1
        else:
            print("Unfortunately your artist was not found. Please try another one.")
            loop = 0
       
    print()
    print()
    #searching for song id
    id_input_song = search_song(title, artist, limit=1)
    
    #print("searching for audio features")
    id_input_song_audio_features = pd.DataFrame(get_single_audio_features(id_input_song))
    id_input_song_audio_features_num = id_input_song_audio_features.select_dtypes(np.number)
    
    #print("apply scaler and umap")
    input_scaled_umap = apply_scale_umap(id_input_song_audio_features_num)
    input_scaled_umap_df = pd.DataFrame(input_scaled_umap[1])
    
    #print("import umap_df and concat to full_df")
    umap_df = pd.read_csv("X_umap_transformed_df_UMAP_HDBSCAN.csv")
    umap_df_new = umap_df.drop(columns = ['Unnamed: 0','cluster'])
    full_df = pd.concat([full_df,umap_df],axis = 1)
    full_df.drop(columns = 'Unnamed: 0', inplace = True)
   
    #print("computing distance_matrix and closest distance")
    from scipy.spatial import distance_matrix
    distance = distance_matrix(input_scaled_umap_df, umap_df_new, p=2, threshold=1000000)
    closest_distance_index = np.argmin(distance)
    
    #print("define cluster of the imported song")
    input_cluster = cluster_df.iloc[closest_distance_index]
    input_cluster = input_cluster.values[2]

    
    #checking if the song is in the Hot Song list and print 5 samples
    hot_df = full_df[(full_df['dataset']=='hot')]
    
    print(" ___|)________________________________________________________)")
    sleep(1)
    print("|___/____________________________|___________________________||")
    sleep(0.9)
    print("|__/|_______/|____/|_____/|______|___________________________||")
    sleep(0.8)
    print("|_/(|,\____/_|___/_|____/_|______|___________________________||")
    sleep(0.7)
    print("|_\_|_/___|__|__|__|___|__|___|__|___________________________||")
    sleep(0.6)
    print("|   |     | ()  | ()   | ()   |  |                           ||")
    sleep(0.5)
    print("| (_|   -()-  -()-   -()-   -()- | -()-  -()-  -()-   -()-   ||")
    sleep(0.4)
    print("|________________________________|__|__()_|__()_|__()__|_____||")
    sleep(0.3)
    print("|__/___\_._______________________|__|__|__|__|__|__|___|_____||")
    sleep(0.2)
    print("|__\___|_._______________________|___\_|___\_|___\_|___|_____||")
    sleep(0.1)
    print("|_____/__________________________|____\|____\|____\|_________||")
    sleep(0.1)
    print("|____/___________________________|___________________________||")
    
    print()
    print()
    print()
    
    if id_input_song in hot_df['id']:
        sample_hot = full_df[(full_df['dataset']=='hot')&(full_df['cluster']==input_cluster)].sample(5)
        output = sample_hot[['title','artist','track_link']].reset_index(drop=True, inplace=True)
        print(output.to_string(index=False))

    else:
        sample_not_hot = full_df[(full_df['dataset']=='not_hot')&(full_df['cluster']==input_cluster)].sample(5)
        output = sample_not_hot[['title','artist','track_link']].reset_index(drop=True)
        print(output.to_string(index=False))

    #run the recommender again
    print()
    print()
    question = input("Do you want another recommendation? Yes or No?")
    
    if question.lower() == "yes":
        song_recommender()
    else:
        print()
        print()
        print('Thank you for using our Song-recommender. Have a great Weekend!')
        sleep(3)
        print("   _______       __")
        print(" /   ------.   / ._`_")
        print("|  /         ~--~    \ ")
        print("| |             __    `.____________________ _^-----^ ")
        print("| |  I=|=======/--\=========================| o o o | ")
        print("\ |  I=|=======\__/=========================|_o_o_o_| ")
        print(" \|                   /                       ~    ~ ")
        print("   \       .---.    . ")
        print("     -----'     ~~'' ")

        
        print("Don't forget to keep on rocking!")
        pass
    return      
