import pandas as pd
import numpy as np
from tqdm import tqdm

precision_to_presence = {4:0.95, 3:0.9, 2:0.7, 1:0.4, 0:0.1}

def merge_rounded_within_chunk(chunk, precision):
    chunk['presence'] = 1
    
    # Round coordinates
    chunk['lat_arrondi'] = chunk['lat'].round(precision)
    chunk['lon_arrondi'] = chunk['lon'].round(precision)
    
    # Merge the chunk with itself to find close species
    fusion_chunk = pd.merge(chunk, chunk, on=['lat_arrondi', 'lon_arrondi'], suffixes=('', '_proche'))
    fusion_chunk = fusion_chunk[fusion_chunk['surveyId'] != fusion_chunk['surveyId_proche']]
    fusion_chunk = fusion_chunk[['surveyId', 'lat', 'lon', 'speciesId_proche']]
    fusion_chunk = fusion_chunk.rename(columns={'speciesId_proche': 'speciesId'})
    fusion_chunk['presence'] = precision_to_presence[precision]
    fusion_chunk = fusion_chunk.drop_duplicates(subset=['surveyId', 'speciesId'])
    
    # Concatenate the processed chunk to the all_data DataFrame
    merged_chunk = pd.concat([chunk[['surveyId', 'speciesId', 'lat', 'lon', 'presence']], fusion_chunk], ignore_index=True)
    merged_chunk = merged_chunk.drop_duplicates(subset=['surveyId', 'speciesId'])

    return merged_chunk

def add_round_data_chunked(file_path: str, precision: int, headlines: int = 20000, chunk_size: int = 5000):
    cols = ['surveyId', 'speciesId', 'lat', 'lon']
    output_file = 'up' + str(precision) + '.csv'
    
    # Initialize an empty DataFrame to collect all processed chunks
    all_data = pd.DataFrame(columns=cols + ['presence'])

    # Read the file in chunks
    chunk_iterator = pd.read_csv(file_path, delimiter=";", usecols=cols, chunksize=chunk_size)

    total_processed = 0

    for chunk in tqdm(chunk_iterator):
        if total_processed >= headlines:
            break
        
        # Limit the number of rows in the current chunk to not exceed the headlines limit
        if total_processed + len(chunk) > headlines:
            chunk = chunk.head(headlines - total_processed)

        # Merge rounded data within the chunk
        merged_chunk = merge_rounded_within_chunk(chunk, precision)
        
        # Concatenate the processed chunk to the all_data DataFrame
        all_data = pd.concat([all_data, merged_chunk], ignore_index=True)

        total_processed += len(chunk)

    # Eliminate duplicates for the same surveyId and speciesId
    all_data = all_data.drop_duplicates(subset=['surveyId', 'speciesId'])

    # Sort the DataFrame by 'surveyId'
    all_data = all_data.sort_values(by='surveyId')

    # Save the final DataFrame to a CSV file
    all_data.to_csv(output_file, index=False)

    return all_data

def merge_rounded_datasets(df_1: pd.DataFrame, df_2: pd.DataFrame):
    # Create indices from 'surveyId' and 'speciesId' columns for both DataFrames
    df_1_indexed = df_1.set_index(['surveyId', 'speciesId'])
    df_2_indexed = df_2.set_index(['surveyId', 'speciesId'])

    # Identify rows present in df_2 but not in df_1
    missing_index = df_2_indexed.index.difference(df_1_indexed.index)

    # Add missing rows from df_2 to df_1
    df_1_updated = pd.concat([df_1, df_2[df_2.set_index(['surveyId', 'speciesId']).index.isin(missing_index)]], ignore_index=True)
    
    df_1_updated = df_1_updated.sort_values(by='surveyId')

    return df_1_updated

def create_datasets_with_presence(metadata_path : str, headlines : int = 20000, chunk_size: int = 5000):
    
    df_1 = add_round_data_chunked(metadata_path, 1, headlines, chunk_size)
    print("add_round1_ok")
    
    df_2 = add_round_data_chunked(metadata_path, 2, headlines, chunk_size)
    df_3 = add_round_data_chunked(metadata_path, 3, headlines, chunk_size)
    #df_4 = add_round_data_chunked(metadata_path, 4, headlines)
    
    print("add_round_ok")
    
    metadata_df = merge_rounded_datasets(df_3, df_2)
    metadata_df = merge_rounded_datasets(metadata_df, df_1)
    
    print("merge_round_ok")
    
    cols = ['surveyId', 'speciesId', 'lat', 'lon','presence']
    metadata_df = metadata_df[cols]
    metadata_df.to_csv('metadata_for_presence_all.csv', index=False)
    
    return metadata_df

# Define the file path to the metadata
file_path = "/home/dakbarin/data/data/GEOLIFECLEF/GLC24_PA_metadata_train.csv"
metadata_df = create_datasets_with_presence(file_path, 1400000, 700000)

df_0 = pd.read_csv('metadata_for_presence_all.csv')
print((len(df_0)))
filtered_df = df_0[(df_0['presence'] == 0.4)]

# Display the filtered DataFrame
print(filtered_df)
