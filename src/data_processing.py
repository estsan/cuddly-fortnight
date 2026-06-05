from pathlib import Path

import pandas as pd


def load_data() -> pd.DataFrame:
    """Reads the data from the file into a dataframe"""

    _file_path_data = Path(__file__).resolve().parent \
        / 'data' / 'penguins_size.csv'
    data = pd.read_csv(_file_path_data) 
    return data


def process_data(data: pd.DataFrame) -> list[pd.DataFrame]:
    """Function to process data for the specific dataset.
    - Handles null values and wrong values for sex.
    - Encodes text values.
    - Splits data set into features and target.
    """

    data = data.dropna(subset=['sex'])
    data = data[data['sex'] != '.']
    data.rename(columns={'sex': 'female'}, inplace=True)
    data['female'] = data['female'].map({'FEMALE':1, 'MALE':0,})

    species_order = {'Adelie': 0, 'Chinstrap': 1, 'Gentoo': 2}
    data['species_encoded'] = data['species'].map(species_order)
    island_order = {'Biscoe': 0, 'Dream': 1, 'Torgersen': 2}
    data['island_encoded'] = data['island'].map(island_order)

    features = ['culmen_length_mm', 
                'culmen_depth_mm', 
                'flipper_length_mm', 
                'body_mass_g', 
                'female', 
                'island_encoded'
                ]
    target = ['species_encoded']

    return [data[features], data[target]]
