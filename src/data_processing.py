from pathlib import Path

import pandas as pd

class DataProcessor():
    """ Class for loading data and process it """

    def __init__(self):
        _file_path_data = Path(__file__).resolve().parent \
            / 'data' / 'penguins_size.csv'
        self.data = pd.read_csv(_file_path_data)

    def process_data(self) -> list[pd.DataFrame]:
        """Function to process data for the specific dataset.
        - Handles null values and wrong values for sex.
        - Encodes text values.
        - Splits data set into features and target.
        """

        self.data = self.data.dropna(subset=['sex'])
        self.data = self.data[self.data['sex'] != '.']
        self.data.rename(columns={'sex': 'female'}, inplace=True)
        self.data['female'] = self.data['female'].map({'FEMALE':1, 'MALE':0,})

        species_order = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
        self.data['species_encoded'] = self.data['species'].map(species_order)
        island_order = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
        self.data['island_encoded'] = self.data['island'].map(island_order)

        features = ['culmen_length_mm', 
                    'culmen_depth_mm', 
                    'flipper_length_mm', 
                    'body_mass_g', 
                    'female', 
                    'island_encoded'
                    ]
        target = ['species_encoded']

        return [self.data[features], self.data[target]]
