import pandas as pd

def preprocess_lalonde_data() -> pd.DataFrame:
    """
    Load and preprocess the LaLonde dataset.
    
    Returns:
        pd.DataFrame: The processed dataset
    """
    url = "https://raw.githubusercontent.com/robjellis/lalonde/master/lalonde_data.csv"
    df = pd.read_csv(url)
    y = df['re78']
    t = df['treat']
    X = df.drop(columns=['re78', 'treat','ID'])

    df = pd.DataFrame({'y': y, 'treat': t, **X})
    return df 