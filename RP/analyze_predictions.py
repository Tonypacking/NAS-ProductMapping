import pandas as pd
import numpy as np
import os
class Analyzer:

    def __init__(self, predictions_path:str, evo_search_path :str, back_search_path :str):
        self._predictions = pd.read_csv(predictions_path)
        self._metrics = ['f1_score_weighted','f1_score_binary', 'precision', 'recall', 'accuracy','f1_score_macro','f1_score_micro']
        self._predictions.sort_values(by=self._metrics, inplace=True, ascending=False)
        self._evo_path = evo_search_path
        self._back_search_path = back_search_path


    def analyze(self, output_path: str):
        """Analyzes the predictions of trained models and writes the results to a file.

        Args:
            output_path (str): Path to validation csv file.
        """
        training_datasets = np.unique(self._predictions['Train dataset'])

        for trained_dataset in training_datasets:
            with open(f'{os.path.join(output_path, 'Training_dataset_'+trained_dataset+'.txt')}', mode='w') as  file:
                file.write(f"Trained dataset: {trained_dataset}\n\n")

                for reduction in np.unique(self._predictions['Dimension reduction']):
                    file.write(f"Dimension reduction : {reduction}\n")
                    data = self._predictions.sort_values(by=self._metrics, ascending=False)[(self._predictions['Train dataset'] == trained_dataset) & (self._predictions['Dimension reduction'] == reduction)].drop(['Dimension reduction', 'Train dataset'], axis=1)
                    file.write(f"{data.to_string(index=False)}\n\n")


