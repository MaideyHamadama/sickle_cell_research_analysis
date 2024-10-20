# Import tools for the data cleaning
from analysis_func import *

def main():
    """ This python script is to clean the source data for analysis """

    # First build the dataframe
    df = load_data()
    
    # Remove rows with null values in key variables
    df = df.dropna(subset=['CODE', 'AGE', 'SEXE', 'NIVEAUSCOLAIRE', 'STATUTMARITAL', 'ETHNIE', 'AGE_DE_DIAGNOSTIC', 'DECEDES', 'VACCINSAJOUR'])
    
    # Remove rows with null or non numeric values in the code variable
    df['CODE'] = pd.to_numeric(df['CODE'], errors='coerce')
    df = df.dropna(subset=['CODE'])
    
    # Remove rows with age values not in a limit
    min_age = 0
    max_age = 90
    df = df[(df['AGE'] >= min_age) & (df['AGE'] <= max_age)]
    
    # Remove rows with sex not Male or Female, 1 or 2, M or F
    df = df[(df['SEXE'] != 1) | (df['SEXE'] != 2)]
        
    # Remove rows with death status not 1 or 2
    df = df[(df['DECEDES'] != 1.00) | (df['DECEDES'] != 2.00)]
    
    # Remove rows with vaccination status not 1 or 2
    df = df[(df['VACCINSAJOUR'] != 1.00) | (df['VACCINSAJOUR'] != 2.00)]
    
    # Harmonize age at diagnosis
    # Remove "ANS" and "AN" 
    df['AGE_DE_DIAGNOSTIC'] = df['AGE_DE_DIAGNOSTIC'].str.replace("ANS|AN", "", regex=True)
    # Convert elements with "MOIS" to 1 year
    df.loc[df['AGE_DE_DIAGNOSTIC'].str.contains('MOIS'), 'AGE_DE_DIAGNOSTIC'] = 1
    # Replace "INCONNU" and "INCONU" with NaN
    df['AGE_DE_DIAGNOSTIC'] = df['AGE_DE_DIAGNOSTIC'].replace(["INCONNU", "INCONU"], np.nan)
    df.to_excel('data_cleaned.xlsx', index=False)
    
main()