""""
Preprocessing for the GAQ corpus.

Code adapted from Jupyter notebooks. Not optimized for performance and 
very specific to the parts of the GAQ corpus used in the paper.
"""
import argparse
import os
import warnings

import numpy as np
import pandas as pd

# Ignore FutureWarning for downcasting replacement in pandas
warnings.simplefilter(action='ignore', category=FutureWarning)

def load_corpora(path: str) -> pd.DataFrame:
    """Load the debate subsets of the GAQ corpus.
    
    Assumes that the GAQ corpus is stored in a directory with the following
    structure:

    path/
        debate_forums_crowd.csv
        debate_forums_experts.csv
        ...

    Args:
        path (str): Path to the directory containing the GAQ corpus.

    Returns:
        pd.DataFrame, pd.DataFrame:
            DataFrames containing the crowd and expert debate corpora.
    """
    crowd = pd.read_csv(os.path.join(path, 'debate_forums_crowd.csv'))
    experts = pd.read_csv(os.path.join(path, 'debate_forums_experts.csv'))
    return crowd, experts

def adapt_dataframe(df_data: pd.DataFrame,
                    ann_num: int) -> pd.DataFrame:
    '''
    Transforms the df into de desired format.

    Args:
        df_data: The original df to be transformed
        ann_num: The number of annotators for this df

    Returns:
        pd.Dataframe: The transformed df
    '''
    # Add the first columns to the df in the  right order
    df_data_prep = df_data[['id', 'cogency_mean', 'effectiveness_mean',
                            'reasonableness_mean', 'overall_mean',
                            'argumentative_majority', 'text',
                            'title']].copy()

    # Create the other columns and place the values accordingly
    for id, row in df_data.iterrows():
        # Iterate through columns excluding the ones not to be changed
        for col in df_data.columns:
            if col not in ['id', 'cogency_mean', 'effectiveness_mean',
                           'reasonableness_mean', 'overall_mean',
                           'argumentative_majority', 'text', 'title']:
                # Loop over the amount of annotators to create a column
                # for each
                for ann in range(1, ann_num+1):
                    new_col_name = f'{col}_{ann}'

                    # If the column doesnt exist yet create it
                    if new_col_name not in df_data_prep.columns:
                        df_data_prep[new_col_name] = pd.NA

                    # Update the values in the corresponding column
                    string = df_data.loc[id, col][1:-1]
                    split = string.split(', ')
                    #if ann-1 < len(split): #add if you want to take
                    # multiple amounts of annotations (ex: 23 and 17) in
                    # the same df
                    df_data_prep.loc[id, new_col_name] = split[ann-1]

    # Order the columns
    fixed_columns = ['id', 'text', 'title']
    prefix_order = []
    for col in df_data.columns:
       if col not in fixed_columns:
           if col not in prefix_order:
               prefix_order.append(col)

    # Create the order by prefix of the columns in df_data_prep
    order_columns = fixed_columns.copy()
    for prefix in prefix_order:
        for col in df_data_prep.columns:
            if col.startswith(prefix + '_'):
                order_columns.append(col)
    
    # Reorder df_data_prep columns
    df_data_prep = df_data_prep[order_columns]

    return df_data_prep

def preprocess_gaq(crowd: pd.DataFrame,
                   experts: pd.DataFrame) -> pd.DataFrame:
    """Preprocess the GAQ corpus.
    
    Args:
        crowd (pd.DataFrame):
            DataFrame containing the crowd debate corpus.
        experts (pd.DataFrame):
            DataFrame containing the expert debate corpus.

    Returns:
        pd.DataFrame:
            Preprocessed DataFrame containing the crowd debate corpus.
    """
    # Only take the items annotated by 17 annotators
    crowd = crowd.loc[900:]

    # Adapt debate forum datasets
    experts = adapt_dataframe(experts, 3)
    crowd = adapt_dataframe(crowd, 17)

    # rename columns for the join
    crowd.rename(columns={
        'argumentative_majority': 'argumentative_majority_crowd',
        'cogency_mean': 'cogency_mean_crowd',
        'effectiveness_mean': 'effectiveness_mean_crowd',
        'reasonableness_mean': 'reasonableness_mean_crowd',
        'overall_mean': 'overall_quality_mean_crowd'}, inplace=True)
    
    experts.rename(columns={
        'overall_1': 'overall_quality_18',
        'overall_2': 'overall_quality_19',
        'overall_3': 'overall_quality_20',
        'cogency_1': 'cogency_18',
        'cogency_2': 'cogency_19',
        'cogency_3': 'cogency_20',
        'effectiveness_1': 'effectiveness_18',
        'effectiveness_2': 'effectiveness_19',
        'effectiveness_3': 'effectiveness_20',
        'reasonableness_1': 'reasonableness_18',
        'reasonableness_2': 'reasonableness_19',
        'reasonableness_3': 'reasonableness_20',
        'argumentative_1': 'argumentative_18',
        'argumentative_2': 'argumentative_19',
        'argumentative_3': 'argumentative_20',
        'argumentative_majority': 'argumentative_majority_experts',
        'cogency_mean': 'cogency_mean_experts',
        'effectiveness_mean': 'effectiveness_mean_experts',
        'reasonableness_mean': 'reasonableness_mean_experts',
        'overall_mean': 'overall_quality_mean_experts'}, inplace=True)
    
    # Merge the datasets
    df = pd.merge(crowd, experts, on=['id', 'text', 'title'], how='inner')

    # introduce some consistency in naming the dimensions of AQ
    # with Dagstuhl
    df.rename(columns={'overall_1': 'overall_quality_1',
                       'overall_2': 'overall_quality_2',
                       'overall_3': 'overall_quality_3',
                       'overall_4': 'overall_quality_4',
                       'overall_5': 'overall_quality_5',
                       'overall_6': 'overall_quality_6',
                       'overall_7': 'overall_quality_7',
                       'overall_8': 'overall_quality_8',
                       'overall_9': 'overall_quality_9',
                       'overall_10': 'overall_quality_10',
                       'overall_11': 'overall_quality_11',
                       'overall_12': 'overall_quality_12',
                       'overall_13': 'overall_quality_13',
                       'overall_14': 'overall_quality_14',
                       'overall_15': 'overall_quality_15',
                       'overall_16': 'overall_quality_16',
                       'overall_17': 'overall_quality_17',}, inplace=True)
    
    # rename # into NaN & replace other missing value characters in case
    df = df.replace(['\'#\'', '#', '\'n/a\'', 'n/a'], np.nan)

    # all items that have been considered argumentative by at least one person
    df_cleaned_sparse = df[~df['id'].isin(df[
        ((df['argumentative_1']=='0')|(df['argumentative_1'].isnull()))
        &((df['argumentative_2']=='0')|(df['argumentative_2'].isnull()))
        &((df['argumentative_3']=='0')|(df['argumentative_3'].isnull()))
        &((df['argumentative_4']=='0')|(df['argumentative_4'].isnull()))
        &((df['argumentative_5']=='0')|(df['argumentative_5'].isnull()))
        &((df['argumentative_6']=='0')|(df['argumentative_6'].isnull()))
        &((df['argumentative_7']=='0')|(df['argumentative_7'].isnull()))
        &((df['argumentative_8']=='0')|(df['argumentative_8'].isnull()))
        &((df['argumentative_9']=='0')|(df['argumentative_9'].isnull()))
        &((df['argumentative_10']=='0')|(df['argumentative_10'].isnull()))
        &((df['argumentative_11']=='0')|(df['argumentative_11'].isnull()))
        &((df['argumentative_12']=='0')|(df['argumentative_12'].isnull()))
        &((df['argumentative_13']=='0')|(df['argumentative_13'].isnull()))
        &((df['argumentative_14']=='0')|(df['argumentative_14'].isnull()))
        &((df['argumentative_15']=='0')|(df['argumentative_15'].isnull()))
        &((df['argumentative_16']=='0')|(df['argumentative_16'].isnull()))
        &((df['argumentative_17']=='0')|(df['argumentative_17'].isnull()))
        &((df['argumentative_18']=='0')|(df['argumentative_18'].isnull()))
        &((df['argumentative_19']=='0')|(df['argumentative_19'].isnull()))
        &((df['argumentative_20']=='0')|(df['argumentative_20'].isnull()))
        ]['id'])]
    
    # drop unused columns
    df_cleaned_sparse = df_cleaned_sparse.drop(columns=[
        'argumentative_1', 'argumentative_2', 'argumentative_3',
        'argumentative_4', 'argumentative_5',
        'argumentative_6', 'argumentative_7', 'argumentative_8',
        'argumentative_9', 'argumentative_10',
        'argumentative_11', 'argumentative_12',
        'argumentative_13', 'argumentative_14', 'argumentative_15',
        'argumentative_16', 'argumentative_17', 'argumentative_18',
        'argumentative_19', 'argumentative_20',
        'argumentative_majority_experts', 'argumentative_majority_crowd'])

    # build new mean values: mean over all 20 and mean of the two means
    df_cleaned_sparse['cogency_mean_experts'] = df_cleaned_sparse[
        'cogency_mean_experts'].astype('float64')
    df_cleaned_sparse['cogency_mean_crowd'] = df_cleaned_sparse[
        'cogency_mean_crowd'].astype('float64')
    df_cleaned_sparse['cogency_mean_weighted'] = (
        df_cleaned_sparse['cogency_mean_experts'] + \
            df_cleaned_sparse['cogency_mean_crowd'])/2

    num_annotators_crowd = df_cleaned_sparse[[
        'cogency_1', 'cogency_2', 'cogency_3', 'cogency_4', 'cogency_5',
        'cogency_6', 'cogency_7', 'cogency_8', 'cogency_9', 'cogency_10',
        'cogency_11', 'cogency_12', 'cogency_13', 'cogency_14',
        'cogency_15', 'cogency_16', 'cogency_17']].notna().sum(1)
    num_annotators_expert = df_cleaned_sparse[[
        'cogency_18', 'cogency_19', 'cogency_20']].notna().sum(1)

    df_cleaned_sparse['cogency_mean'] = (
        df_cleaned_sparse['cogency_mean_crowd'] * num_annotators_crowd + \
            df_cleaned_sparse['cogency_mean_experts'] * num_annotators_expert
        )/(num_annotators_crowd+num_annotators_expert)

    df_cleaned_sparse['effectiveness_mean_experts'] = df_cleaned_sparse[
        'effectiveness_mean_experts'].astype('float64')
    df_cleaned_sparse['effectiveness_mean_crowd'] = df_cleaned_sparse[
        'effectiveness_mean_crowd'].astype('float64')
    df_cleaned_sparse['effectiveness_mean_weighted'] = (df_cleaned_sparse[
        'effectiveness_mean_experts'] + \
              df_cleaned_sparse['effectiveness_mean_crowd'])/2

    num_annotators_crowd = df_cleaned_sparse[[
        'effectiveness_1', 'effectiveness_2', 'effectiveness_3',
        'effectiveness_4', 'effectiveness_5',
        'effectiveness_6', 'effectiveness_7', 'effectiveness_8',
        'effectiveness_9', 'effectiveness_10',
        'effectiveness_11', 'effectiveness_12', 'effectiveness_13',
        'effectiveness_14', 'effectiveness_15',
        'effectiveness_16', 'effectiveness_17']].notna().sum(1)
    num_annotators_expert = df_cleaned_sparse[[
        'effectiveness_18', 'effectiveness_19', 'effectiveness_20'
        ]].notna().sum(1)

    df_cleaned_sparse['effectiveness_mean'] = (
        df_cleaned_sparse['effectiveness_mean_crowd'] * \
            num_annotators_crowd + \
                df_cleaned_sparse['effectiveness_mean_experts'] * \
                    num_annotators_expert
        )/(num_annotators_crowd+num_annotators_expert)
    
    df_cleaned_sparse['reasonableness_mean_experts'] = df_cleaned_sparse[
        'reasonableness_mean_experts'].astype('float64')
    df_cleaned_sparse['reasonableness_mean_crowd'] = df_cleaned_sparse[
        'reasonableness_mean_crowd'].astype('float64')
    df_cleaned_sparse['reasonableness_mean_weighted'] = (df_cleaned_sparse[
        'reasonableness_mean_experts'] + \
            df_cleaned_sparse['reasonableness_mean_crowd'])/2

    num_annotators_crowd = df_cleaned_sparse[[
        'reasonableness_1', 'reasonableness_2', 'reasonableness_3',
        'reasonableness_4', 'reasonableness_5',
        'reasonableness_6', 'reasonableness_7', 'reasonableness_8',
        'reasonableness_9', 'reasonableness_10',
        'reasonableness_11', 'reasonableness_12', 'reasonableness_13',
        'reasonableness_14', 'reasonableness_15',
        'reasonableness_16', 'reasonableness_17']].notna().sum(1)
    num_annotators_expert = df_cleaned_sparse[[
        'reasonableness_18', 'reasonableness_19', 'reasonableness_20']]. \
            notna().sum(1)

    df_cleaned_sparse['reasonableness_mean'] = (
        df_cleaned_sparse['reasonableness_mean_crowd'] * \
            num_annotators_crowd + \
                df_cleaned_sparse['reasonableness_mean_experts'] * \
                    num_annotators_expert
        )/(num_annotators_crowd+num_annotators_expert)
    
    df_cleaned_sparse['overall_quality_mean_experts'] = df_cleaned_sparse[
        'overall_quality_mean_experts'].astype('float64')
    df_cleaned_sparse['overall_quality_mean_crowd'] = df_cleaned_sparse[
        'overall_quality_mean_crowd'].astype('float64')
    df_cleaned_sparse['overall_quality_mean_weighted'] = (
        df_cleaned_sparse['overall_quality_mean_experts'] + \
            df_cleaned_sparse['overall_quality_mean_crowd']) / 2

    num_annotators_crowd = df_cleaned_sparse[[
        'overall_quality_1', 'overall_quality_2', 'overall_quality_3',
        'overall_quality_4', 'overall_quality_5',
        'overall_quality_6', 'overall_quality_7', 'overall_quality_8',
        'overall_quality_9', 'overall_quality_10',
        'overall_quality_11', 'overall_quality_12', 'overall_quality_13',
        'overall_quality_14', 'overall_quality_15',
        'overall_quality_16', 'overall_quality_17']].notna().sum(1)
    num_annotators_expert = df_cleaned_sparse[[
        'overall_quality_18',
        'overall_quality_19',
        'overall_quality_20']].notna().sum(1)

    df_cleaned_sparse['overall_quality_mean'] = (
        df_cleaned_sparse['overall_quality_mean_crowd'] * \
            num_annotators_crowd + \
                df_cleaned_sparse['overall_quality_mean_experts'] * \
                    num_annotators_expert
        )/(num_annotators_crowd+num_annotators_expert)
    
    return df_cleaned_sparse

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--path', type=str, required=True,
                        help="Path to the directory containing the GAQ"
                        " corpus.")
    parser.add_argument('--output', type=str, required=True,
                        help="Path to the output file.")
    args = parser.parse_args()

    crowd, experts = load_corpora(args.path)
    df = preprocess_gaq(crowd, experts)
    df.to_csv('gaq.csv', index=False)
    print(f"Preprocessed GAQ corpus saved to {args.output}.")
