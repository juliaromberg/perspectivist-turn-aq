"""
Preprocessing of the Dagstuhl corpora.

Code adapted from Jupyter notebooks. Not optimized for
performance and very specific to the Dagstuhl corpora used in the paper.
"""
import argparse
import json
import os
import re
import warnings

import numpy as np
import pandas as pd

# Ignore performance warnings for pandas
warnings.simplefilter(action='ignore',
                      category=pd.errors.PerformanceWarning)

def majority(column_values: pd.Series) -> int:
    """
    Calculate the majority voting for a column.

    Args:
        column_values: The values of the column.

    Returns:
        int: The majority voting value.
    """
    if column_values.isna().all():
        return pd.NA
    
    column_values = column_values.dropna()

    if (column_values.value_counts() == 1).all():
        return 2
    try:
        return column_values.mode()[0]
    except IndexError:
        return pd.NA
    
def load_data(data_path: str,
              annotation_type: str) -> pd.DataFrame:
    """
    Load the data from the given path.

    Args:
        data_path: The path to the data.
        annotation_type: The type of the annotation.

    Returns:
        pd.DataFrame: The loaded data.
    """
    if annotation_type == "expert":
        sep = '\t'
    elif annotation_type == "crowd":
        sep = ','
    data = pd.read_csv(data_path, sep=sep,
                       encoding='ISO-8859-1')
    return data

def process_expert_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Process the data.

    Args:
        df: The data to process.

    Returns:
        pd.DataFrame: The processed data.
    """
    # Rename sufficiency to local_sufficiency for clarity.
    df.rename(columns={'sufficiency': 'local sufficiency'},
                   inplace=True)
    # Add the 3 first columns to the df in the  right order
    df_data_prep = df[['#id', 'argument', 'issue', 'stance']].copy()
    # Drop duplicate arguments
    df_data_prep = df_data_prep.drop_duplicates(subset=['#id'])

    # Create the other columns and place the values accordingly
    for _, row in df.iterrows():
        row_id = row['#id']
        suffix = row['annotator']

        # Iterate through columns excluding '#id', 'annotator',
        # 'argument', 'issue' and 'stance'
        for col in df.columns:
            if col not in ['#id', 'annotator', 'argument',
                           'issue', 'stance']:
                new_col_name = f"{col}_{suffix}"

                # Check if the column exists, if not create it
                if new_col_name not in df_data_prep.columns:
                    df_data_prep[new_col_name] = pd.NA

                # Update the value in the corresponding column
                df_data_prep.loc[df_data_prep['#id'] == row_id,
                                 new_col_name] = row[col]

    # Order the columns
    fixed_columns = ['#id', 'argument', 'issue', 'stance']
    prefix_order = []
    for col in df.columns:
        if col not in fixed_columns:
            if col not in prefix_order:
                prefix_order.append(col)

    # Create a list of columns in df_data_prep ordered by the prefix
    ordered_columns = fixed_columns.copy()
    for prefix in prefix_order:
        for col in df_data_prep.columns:
            if col.startswith(prefix + '_'):
                ordered_columns.append(col)

    # Reorder df_data_prep columns
    df_data_prep = df_data_prep[ordered_columns]

    # Loop over the column names
    df_data_prep.columns = [col.replace(' ', '_') 
                            for col
                            in df_data_prep.columns]
    
    # Set the one "Cannot judge" value to NA
    df_data_prep.loc[df_data_prep['#id'] == \
                     'arg240623', 'global_relevance_3'] = pd.NA
    
    # Keep just the number in the qulity columns
    # Function to remove text after the space
    def remove_after_space(x):
      if isinstance(x, str):
        return x.split(' ', 1)[0]
      return x

    # Remove str part
    columns_to_modify = df_data_prep. \
        loc[:, 'overall_quality_1':'local_sufficiency_3']
    df_data_prep[columns_to_modify.columns] = columns_to_modify. \
        map(remove_after_space)

    # add mean and majority columns
    prefixes = set('_'.join(col.split('_')[:-1])
                   for col
                   in df_data_prep.columns
                   if col not in ['#id', 'argument', 'issue', 'stance'])

    df_data_final = df_data_prep.copy()
    for prefix in prefixes:
        cols = [col
                for col
                in df_data_final.columns
                if col.startswith(prefix)]
        if prefix != 'argumentative':
            df_data_final[cols] = df_data_final[cols].apply(pd.to_numeric)
            df_data_final[f'{prefix}_mean'] = df_data_final[cols].mean(axis=1)
        df_data_final[f'{prefix}_majority'] = df_data_final[cols].apply(majority, axis=1)

    # Order the columns
    fixed_columns = ['#id', 'argument', 'issue', 'stance']
    prefix_order = sorted(prefixes)

    # Create a list of columns in df_data_prep ordered by the prefix
    ordered_columns = fixed_columns.copy()
    for prefix in prefix_order:
        for col in df_data_final.columns:
            if col.startswith(prefix + '_'):
                ordered_columns.append(col)

    # Reorder df_data_prep columns
    df = df_data_final[ordered_columns]

    # Drop the rows that are not considered argumentative by any annotator.
    df_cleaned_sparse = df[~df['#id'].isin(
        df[(df['argumentative_1']=='n') & \
           (df['argumentative_2']=='n') & \
            (df['argumentative_3']=='n')]['#id'])]
    df_cleaned_dense = df[df['#id'].isin(
        df[(df['argumentative_1']=='y') & \
           (df['argumentative_2']=='y') & \
            (df['argumentative_3']=='y')]['#id'])]
    
    # drop unused columns
    df_cleaned_sparse = df_cleaned_sparse.drop(
        columns=['argumentative_1', 'argumentative_2', 'argumentative_3',
                 'argumentative_majority'])
    df_cleaned_dense = df_cleaned_dense.drop(
        columns=['argumentative_1', 'argumentative_2', 'argumentative_3',
                 'argumentative_majority'])

    return df_cleaned_sparse, df_cleaned_dense

def preprocess_crowd_data(df: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the crowd data.

    Args:
        df: The data to preprocess.

    Returns:
        pd.DataFrame: The preprocessed data.
    """
    # Drop unnecessary Columns
    df_temp = df.drop(['_unit_id', '_created_at', '_id', '_started_at',
                       '_tainted', '_channel', '_trust', '_country',
                       '_region', '_city', '_ip', 
                       '_please_provide_us_with_anything_else_that_affected_your_assessment_of_the_arguments_and_that_is_not_covered_by_the_above_dimensions', 
                       'how_do_you_rate_the_overall_quality_of_the_authors_argumentation_gold', 
                       'how_would_you_rate_the_acceptability_of_the_premises_of_the_authors_arguments_gold', 
                       'how_would_you_rate_the_appropriateness_of_the_style_of_the_authors_argumentation_gold', 
                       'how_would_you_rate_the_arrangement_of_the_authors_argumentation_gold', 
                       'how_would_you_rate_the_clarity_of_the_style_of_the_authors_argumentation_gold', 
                       'how_would_you_rate_the_cogency_of_the_authors_arguments__gold', 
                       'how_would_you_rate_the_effectiveness_of_the_authors_argumentation_gold', 
                       'how_would_you_rate_the_global_acceptability_of_the_authors_argumentation_gold', 
                       'how_would_you_rate_the_global_relevance_of_the_authors_argumentation_gold', 
                       'how_would_you_rate_the_global_sufficiency_of_the_authors_argumentation_gold', 
                       'how_would_you_rate_the_reasonableness_of_the_authors_argumentation_gold', 
                       'how_would_you_rate_the_relevance_of_the_premises_to_the_authors_conclusions__gold', 
                       'how_would_you_rate_the_success_of_the_authors_argumentation_in_creating_credibility_gold', 
                       'how_would_you_rate_the_success_of_the_authors_argumentation_in_making_an_emotional_appeal__gold', 
                       'how_would_you_rate_the_sufficiency_of_the_premises_of_the_authors_arguments__gold'], axis=1)
    
    # Re-order the rest of the columns
    df_temp = df_temp[['_worker_id', 'exp_id', 'content', 'issue',
                       'stance',
                       'how_do_you_rate_the_overall_quality_of_the_authors_argumentation', 
                       'how_would_you_rate_the_acceptability_of_the_premises_of_the_authors_arguments', 
                       'how_would_you_rate_the_appropriateness_of_the_style_of_the_authors_argumentation', 
                       'how_would_you_rate_the_arrangement_of_the_authors_argumentation', 
                       'how_would_you_rate_the_clarity_of_the_style_of_the_authors_argumentation',
                       'how_would_you_rate_the_cogency_of_the_authors_arguments_', 
                       'how_would_you_rate_the_effectiveness_of_the_authors_argumentation', 
                       'how_would_you_rate_the_global_acceptability_of_the_authors_argumentation', 
                       'how_would_you_rate_the_global_relevance_of_the_authors_argumentation', 
                       'how_would_you_rate_the_global_sufficiency_of_the_authors_argumentation', 
                       'how_would_you_rate_the_reasonableness_of_the_authors_argumentation', 
                       'how_would_you_rate_the_relevance_of_the_premises_to_the_authors_conclusions_', 
                       'how_would_you_rate_the_success_of_the_authors_argumentation_in_creating_credibility', 
                       'how_would_you_rate_the_success_of_the_authors_argumentation_in_making_an_emotional_appeal_', 
                       'how_would_you_rate_the_sufficiency_of_the_premises_of_the_authors_arguments_']]
    
    # Rename those same columns
    df_temp.columns = ['ann_id', '#id', 'argument', 'issue', 'stance',
                       'overall_quality', 'local_acceptability',
                       'appropriateness', 'arrangement', 'clarity',
                       'cogency', 'effectiveness', 'global_acceptability',
                       'global_relevance', 'global_sufficiency',
                       'reasonableness', 'local_relevance', 'credibility', 
                       'emotional_appeal', 'local_sufficiency']
    
    # Change annotator IDs to sequential small indegers
    # Create a mapping of unique values to ordered numbers
    unique_values = df_temp['ann_id'].unique()
    mapping = {val: idx + 1 for idx, val in enumerate(unique_values)}

    # Create a new column 'changed_A' by applying the mapping
    df_temp.loc[:, 'annotator'] = df_temp['ann_id'].map(mapping)

    # drop ann_id column
    df_temp = df_temp.drop(['ann_id'], axis=1)

    # Adapt dataframe
    # Add the 3 first columns to the df in the  right order
    df_data_prep = df_temp[['#id', 'argument', 'issue', 'stance']].copy()
    # Drop duplicate arguments
    df_data_prep = df_data_prep.drop_duplicates(subset=['#id'])

    # Create the other columns and place the values accordingly
    for _, row in df_temp.iterrows():
        row_id = row['#id']
        suffix = row['annotator']   
        # Iterate through columns excluding '#id', 'annotator',
        # 'argument', 'issue' and 'stance'
        for col in df_temp.columns:
            if col not in ['#id', 'annotator', 'argument', 'issue', 'stance']:
                new_col_name = f"{col}_{suffix}"    
                # Check if the column exists, if not create it
                if new_col_name not in df_data_prep.columns:
                    df_data_prep[new_col_name] = pd.NA    
                    # Update the value in the corresponding column
                df_data_prep.loc[df_data_prep['#id'] == row_id,
                                 new_col_name] = row[col]

    # Order the columns
    fixed_columns = ['#id', 'argument', 'issue', 'stance']
    prefix_order = []
    for col in df_temp.columns:
        if col not in fixed_columns:
            if col not in prefix_order:
                prefix_order.append(col)

    # Create a list of columns in df_data_prep ordered by the prefix
    ordered_columns = fixed_columns.copy()
    for prefix in prefix_order:
        for col in df_data_prep.columns:
            if col.startswith(prefix + '_'):
                ordered_columns.append(col)

    # Reorder df_data_prep columns
    df_data_prep = df_data_prep[ordered_columns]

    # Function to remove text after the space
    def remove_after_space(x):
        if isinstance(x, str):
            return x.split(' ', 1)[0]
        return x

    # Remove str part
    columns_to_modify = df_data_prep.loc[:, 'overall_quality_1':'local_sufficiency_3']
    df_data_prep[columns_to_modify.columns] = columns_to_modify.map(remove_after_space)
    
    prefixes = set('_'.join(col.split('_')[:-1]) for col
                   in df_data_prep.columns
                   if col not in ['#id', 'argument', 'issue', 'stance'])

    df_data_final = df_data_prep.copy()
    for prefix in prefixes:
        cols = [col for col in df_data_final.columns if col.startswith(prefix)]
        df_data_final[cols] = df_data_final[cols].apply(pd.to_numeric, errors='coerce')
        df_data_final[f'{prefix}_mean'] = df_data_final[cols].mean(axis=1)
        df_data_final[f'{prefix}_majority'] = df_data_final[cols].apply(majority, axis=1)

    # Order the columns
    fixed_columns = ['#id', 'argument', 'issue', 'stance']
    prefix_order = sorted(prefixes)

    # Create a list of columns in df_data_prep ordered by the prefix
    ordered_columns = fixed_columns.copy()
    for prefix in prefix_order:
        for col in df_data_final.columns:
            if col.startswith(prefix + '_'):
                ordered_columns.append(col)

    # Reorder df_data_prep columns
    df_data_final = df_data_final[ordered_columns]

    return df_data_final

def preprocess_novice_data(path):
    """
    load and preprocess novice data
    """
    # list all files in the directory
    files = os.listdir(path)

    # get the group and member identifiers
    # filenames have structure 
    # argquality23-groupX-memberY-DATE.csv
    group_member_identifiers = []
    for f in files:
        group_member_identifiers.append(
            re.findall(r'group(\d+)-member(\d+)', f))

    group_member_identifiers = [item
                                for sublist
                                in group_member_identifiers
                                for item in sublist]

    annotator_identifiers = {}
    annotator_id = 1
    for group, member in group_member_identifiers:
        if (group, member) not in annotator_identifiers:
            annotator_identifiers[(group, member)] = annotator_id
            annotator_id += 1

    annotator_files = {}
    for f in files:
        group, member = re.findall(r'group(\d+)-member(\d+)', f)[0]
        annotator_files[f] = annotator_identifiers[(group, member)]

    annotator_files

    num_annotators = len(files)

    annotations = {}
    for f, idx in annotator_files.items():
        annotation = json.load(open(os.path.join(path,f)))
        for k, v in annotation.items():
            arg_id = k.split("-")[0]
            dimension = "_".join(k.split("-")[1:])
            if arg_id not in annotations.keys():
                annotations[arg_id] = {dimension + f"_{idx}": v}
            else:
                annotations[arg_id][dimension + f"_{idx}"] = v

    df = pd.DataFrame.from_dict(annotations, orient='index')

    dimensions = [c.replace("_1", "") for c in df.columns if c.endswith("_1")]

    for d in dimensions:
        if d + "_26" not in df.columns:
            df[d + "_26"] = [np.nan for _ in range(len(df))]

    df = df.replace("?", np.nan)
    # use 100 as a placeholder for missing values to avoid odd behavior
    # in the calculations of the means
    df = df.fillna(100)

    df["#id"] = df.index

    for d in dimensions:
        mean = []
        majority = []
        for i, row in df.iterrows():
            values = [row[d + f"_{j}"] for j in range(1, num_annotators + 1)]
            values = [int(v) for v in values if v != 100]
            mean.append(np.mean([int(v) for v in values]))
            # calculating majority vote by taking the most common value
            # that is not nan
            majority.append(max(set(values), key = values.count))
            
        df[d + "_mean"] = mean
        df[d + "_majority"] = majority

    # saving the dataframe in the right order
    column_order = ["#id"]
    for d in dimensions:
        for idx in range(1, num_annotators + 1):
            if d + f"_{idx}" in df.columns:
                # some dimensions might not have been annotated by all 
                # annotators
                column_order.append(d + f"_{idx}")
        column_order.append(d + "_mean")
        column_order.append(d + "_majority")

    df = df[column_order]

    # remove the placeholder value
    df = df.replace(100, np.nan)

    return df

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--path", type=str, required=True,
                        help="Path to the data csv for crowd and expert,"
                        "to the data directory for the novice annotations.")
    parser.add_argument("--output", type=str, required=True,
                        help="Path to the output directory.")
    parser.add_argument("--annotation_type", type=str, required=True,
                        help="Type of the annotations"
                        "(expert, crowd, novice).")
    args = parser.parse_args()

    if args.annotation_type == "expert":
        data = load_data(args.path, args.annotation_type)
        sparse, dense = process_expert_data(data)
        sparse.to_csv(os.path.join(args.output, "dagstuhl_expert_sparse.csv"),
                      index=False)
        dense.to_csv(os.path.join(args.output, "dagstuhl_expert_dense.csv"),
                     index=False)
        
    elif args.annotation_type == "crowd":
        data = load_data(args.path, args.annotation_type)
        data = preprocess_crowd_data(data)
        data.to_csv(os.path.join(args.output, "dagstuhl_crowd.csv"),
                    index=False)
        
    elif args.annotation_type == "novice":
        data = preprocess_novice_data(args.path)
        data.to_csv(os.path.join(args.output, "dagstuhl_novice.csv"),
                    index=False)
        
    else:
        raise ValueError("Invalid annotation type."
                         "Choose from expert, crowd, novice.")
    
    print(f"Saved preprocessed data to {args.output}.")

