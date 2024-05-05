import pandas as pd
import argparse
from prompt import work

if __name__ == "__main__":
    # Initialize the parser
    parser = argparse.ArgumentParser(description='Experiment script.')
    # Adding the arguments
    parser.add_argument('--dataset', type=str, required=True, default='UCMerced',
                        help='The dataset to use')
    parser.add_argument('--model', type=str, required=False, default='Gemini1.5',
                        help='The model to use')
    parser.add_argument('--location', type=str, required=False, default='us-central1',
                        help='The location for the experiment')
    parser.add_argument('--num_shot_per_class', type=int, required=True,
                        help='The number of shots per class')
    parser.add_argument('--num_qns_per_round', type=int, required=False, default=1,
                        help='The number of questions asked each time')
    
    # Parsing the arguments
    args = parser.parse_args()

    # Using the arguments
    dataset_name = args.dataset
    model = args.model
    location = args.location
    num_shot_per_class = args.num_shot_per_class
    num_qns_per_round = args.num_qns_per_round
    
    # Folder to load the images, and this will be prepended to the filename stored in the index column of the dataframe.
    IMAGE_FOLDER = f'ManyICL/dataset/{dataset_name}/images'
    
    # Read the two dataframes for the dataset
    demo_df = pd.read_csv(f'ManyICL/dataset/{dataset_name}/demo.csv', index_col=0)
    test_df = pd.read_csv(f'ManyICL/dataset/{dataset_name}/test.csv', index_col=0)
    
    
    classes = list(demo_df.columns)
    class_desp = classes # The actual list of options given to the model. If the column names are informative enough, we can just use them.
    class_to_idx = {class_name: idx for idx, class_name in enumerate(classes)}
    
    work(model, num_shot_per_class, location, num_qns_per_round, test_df, demo_df, classes, class_desp, IMAGE_FOLDER, dataset_name)
