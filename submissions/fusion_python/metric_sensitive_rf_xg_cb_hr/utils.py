import pandas as pd
import numpy as np
from fastcore.basics import Path
            

def tabularize_data(data_dir, feature_cols, ground_truth=None, lag_steps=1, fill_na=True):
    merged_data = pd.DataFrame()
    test_data = Path(data_dir).glob('*.csv')
    # Check if test_data is empty
    if not test_data:
        raise ValueError(f'No csv files found in {data_dir}')
    for data_file in test_data:
        data_df = pd.read_csv(data_file)
        data_df['ObjectID'] = int(data_file.stem)
        data_df['TimeIndex'] = range(len(data_df))
    
        lagged_features = []
        new_feature_cols = list(feature_cols)  # Create a copy of feature_cols
        # Create lagged features for each column in feature_cols
        for col in feature_cols:
            for i in range(1, lag_steps+1):
                lag_col_name = f'{col}_lag_{i}'
                data_df[lag_col_name] = data_df.groupby('ObjectID')[col].shift(i)
                new_feature_cols.append(lag_col_name)  # Add the lagged feature to new_feature_cols
        
        # Add the lagged features to the DataFrame all at once
        data_df = pd.concat([data_df] + lagged_features, axis=1)

        if ground_truth is None:
            merged_df = data_df
        else:
            ground_truth_object = ground_truth[ground_truth['ObjectID'] == data_df['ObjectID'][0]].copy()
            # Separate the 'EW' and 'NS' types in the ground truth
            ground_truth_EW = ground_truth_object[ground_truth_object['Direction'] == 'EW'].copy()
            ground_truth_NS = ground_truth_object[ground_truth_object['Direction'] == 'NS'].copy()
            
            # Create 'EW' and 'NS' labels and fill 'unknown' values
            ground_truth_EW['EW'] = ground_truth_EW['Node'] + '-' + ground_truth_EW['Type']
            ground_truth_NS['NS'] = ground_truth_NS['Node'] + '-' + ground_truth_NS['Type']
            ground_truth_EW.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)
            ground_truth_NS.drop(['Node', 'Type', 'Direction'], axis=1, inplace=True)

            # Merge the input data with the ground truth
            merged_df = pd.merge(data_df, 
                                ground_truth_EW.sort_values('TimeIndex'), 
                                on=['TimeIndex', 'ObjectID'],
                                how='left')
            merged_df = pd.merge_ordered(merged_df, 
                                        ground_truth_NS.sort_values('TimeIndex'), 
                                        on=['TimeIndex', 'ObjectID'],
                                        how='left')

            # Fill 'unknown' values in 'EW' and 'NS' columns that come before the first valid observation
            merged_df['EW'].ffill(inplace=True, limit=6)
            merged_df['EW'].bfill(inplace=True, limit=6)
            merged_df['NS'].ffill(inplace=True, limit=6)
            merged_df['NS'].bfill(inplace=True, limit=6)
            merged_df['EW'] = merged_df['EW'].fillna('XX-XX')
            merged_df['NS'] = merged_df['NS'].fillna('XX-XX')
            
        merged_data = pd.concat([merged_data, merged_df])

    # Fill missing values (for the lagged features)
    if fill_na:
        merged_data.bfill(inplace=True)
    
    return merged_data, new_feature_cols

def convert_classifier_output_sep(ew_df,ns_df):
    # Split the 'Predicted_EW' and 'Predicted_NS' columns into 
    # 'Node' and 'Type' columns
    # ew_df = classifier_output[['TimeIndex', 'ObjectID', 'Predicted_EW']].copy()
    ew_df[['Node', 'Type']] = ew_df['Predicted_EW'].str.split('-', expand=True)
    ew_df['Direction'] = 'EW'
    ew_df.drop(columns=['Predicted_EW'], inplace=True)

    # ns_df = classifier_output[['TimeIndex', 'ObjectID', 'Predicted_NS']].copy()
    ns_df[['Node', 'Type']] = ns_df['Predicted_NS'].str.split('-', expand=True)
    ns_df['Direction'] = 'NS'
    ns_df.drop(columns=['Predicted_NS'], inplace=True)

    # Concatenate the processed EW and NS dataframes
    final_df = pd.concat([ew_df, ns_df], ignore_index=True)

    # Sort dataframe based on 'ObjectID', 'Direction' and 'TimeIndex'
    final_df.sort_values(['ObjectID', 'Direction', 'TimeIndex'], inplace=True)

    # Apply the function to each group of rows with the same 'ObjectID' and 'Direction'
    groups = final_df.groupby(['ObjectID', 'Direction'])
    keep = groups[['Node', 'Type']].apply(lambda group: group.shift() != group).any(axis=1)

    # Filter the DataFrame to keep only the rows we're interested in
    keep.index = final_df.index
    final_df = final_df[keep]

    # Reset the index and reorder the columns
    final_df = final_df.reset_index(drop=True)
    final_df = final_df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']]
    final_df = final_df.sort_values(['ObjectID', 'TimeIndex', 'Direction'])

    return final_df

def convert_classifier_output(classifier_output):
    # Split the 'Predicted_EW' and 'Predicted_NS' columns into 
    # 'Node' and 'Type' columns
    ew_df = classifier_output[['TimeIndex', 'ObjectID', 'Predicted_EW']].copy()
    ew_df[['Node', 'Type']] = ew_df['Predicted_EW'].str.split('-', expand=True)
    ew_df['Direction'] = 'EW'
    ew_df.drop(columns=['Predicted_EW'], inplace=True)

    ns_df = classifier_output[['TimeIndex', 'ObjectID', 'Predicted_NS']].copy()
    ns_df[['Node', 'Type']] = ns_df['Predicted_NS'].str.split('-', expand=True)
    ns_df['Direction'] = 'NS'
    ns_df.drop(columns=['Predicted_NS'], inplace=True)

    # Concatenate the processed EW and NS dataframes
    final_df = pd.concat([ew_df, ns_df], ignore_index=True)

    # Sort dataframe based on 'ObjectID', 'Direction' and 'TimeIndex'
    final_df.sort_values(['ObjectID', 'Direction', 'TimeIndex'], inplace=True)

    # Apply the function to each group of rows with the same 'ObjectID' and 'Direction'
    groups = final_df.groupby(['ObjectID', 'Direction'])
    keep = groups[['Node', 'Type']].apply(lambda group: group.shift() != group).any(axis=1)

    # Filter the DataFrame to keep only the rows we're interested in
    keep.index = final_df.index
    final_df = final_df[keep]

    # Reset the index and reorder the columns
    final_df = final_df.reset_index(drop=True)
    final_df = final_df[['ObjectID', 'TimeIndex', 'Direction', 'Node', 'Type']]
    final_df = final_df.sort_values(['ObjectID', 'TimeIndex', 'Direction'])

    return final_df
