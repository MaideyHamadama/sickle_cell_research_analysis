from analysis_func import *
from lifelines import KaplanMeierFitter
from lifelines.statistics import logrank_test, multivariate_logrank_test

def run():
    # Charging analysis file
    df = load_cleaned_data()

    # Filter key variables
    df = df[['CODE', 'AGE', 'AGE_DE_DIAGNOSTIC', 'DECEDES', 'VACCINSAJOUR',
             'AGEAUDECES', 'TAUXHB', 'HYDREA']]
    
    # Replace 2 by 0 and leave out the 1 in the death variable
    df['DECEDES'] = df['DECEDES'].replace(2,0)
    
    # Lifeline curve
    df['AGEAUDECES'] = pd.to_numeric(df['AGEAUDECES'], errors='coerce')
    df = df.dropna(subset=['AGEAUDECES'])
    
    # Adding new variable timeline
    df['timeline'] = np.where(df['AGE_DE_DIAGNOSTIC'].isnull(), df['AGE'], df['AGE'] - df['AGE_DE_DIAGNOSTIC'])
    
    # Kaplan-Meier Fitter
    kmf = KaplanMeierFitter()
    
    """ LIFETIME CURVE IN FUNCTION OF VACCINATION STATUS """
    # Patients seperation
    vaccinated = df[df['VACCINSAJOUR'] == 1]
    not_vaccinated = df[df['VACCINSAJOUR'] == 2]
    
    
    # Ajustement et tracer pour les patients vaccines
    kmf.fit(vaccinated['timeline'], event_observed=vaccinated['DECEDES'], label='Vaccinated')
    ax = kmf.plot_survival_function()
    
    # Ajustement et tracer pour les patients non vaccines
    kmf.fit(not_vaccinated['timeline'], event_observed=not_vaccinated['DECEDES'], label='Not vaccinated')
    kmf.plot_survival_function(ax=ax)
    
    plt.title("Survival curve by vaccination status")
    plt.show()
    
    # Test du log-rank
    results = logrank_test(vaccinated['timeline'], not_vaccinated['timeline'], event_observed_A=vaccinated['DECEDES'], event_observed_B=not_vaccinated['DECEDES'])
    print(results.summary)
    
    """ LIFETIME CURVE IN FUNCTION OF AGE SLICES """
    df['Age_Group'] = pd.cut(df['AGE'], bins=[20,34,54,70], labels=['21-35','35-55','>55'])
    
    # Tracer la courbe pour chaque tranche d'age
    fig, ax = plt.subplots()
    for label, grouped_df in df.groupby('Age_Group'):
        kmf.fit(grouped_df['timeline'], event_observed=grouped_df['DECEDES'], label=label)
        kmf.plot_survival_function(ax=ax)
        
    plt.title('Survival Curves by Age Group')
    plt.show()
    
    #Test du log-rank
    result = multivariate_logrank_test(df['timeline'], df['Age_Group'], df['DECEDES'])
    print("Global logrank test result: ", result.test_statistic, 'p-value : ', result.p_value)

    """ LIFETIME CURVE IN FUNCITON OF HEMOGLOBINE RATE """

    df['Hb_Rate_Slice'] = pd.cut(df['TAUXHB'], bins=[0,4,8,12], labels=['0-4.0','4.0-8.0','8.0-12.0'])
    
    # Tracer la courbe pour chaque tranche d'age
    fig, ax = plt.subplots()
    for label, grouped_df in df.groupby('Hb_Rate_Slice'):
        kmf.fit(grouped_df['timeline'], event_observed=grouped_df['DECEDES'], label=label)
        kmf.plot_survival_function(ax=ax)
        
    plt.title('Survival Curves by Hb Rate Slice')
    plt.show()
    
    # Test du log-rank
    result = multivariate_logrank_test(df['timeline'], df['Hb_Rate_Slice'], df['DECEDES'])
    print("Global logrank test result: ", result.test_statistic, 'p-value : ', result.p_value)
    
    """ LIFETIME CURVE IN FUNCITON OF HYDROXYUREA PRESCRIPTION AND ADMINISTRATION """

    # Separation des groupes
    admited = df[df['HYDREA'] == 1]
    not_admited = df[df['HYDREA'] == 2]
    
    # Ajustement et tracer pour les patients admis
    kmf.fit(admited['timeline'], event_observed=admited['DECEDES'], label='Hydrea_admis')
    ax = kmf.plot_survival_function()
    
    # Ajustement et tracer pour les patients non admis
    kmf.fit(not_admited['timeline'], event_observed=not_admited['DECEDES'], label='Hydrea_non_admis')
    kmf.plot_survival_function(ax=ax)
    
    plt.title("Survival curve by administration of hydroxyurea status")
    plt.show()
    
    # Test du log-rank
    results = logrank_test(admited['timeline'], not_admited['timeline'], event_observed_A=admited['DECEDES'], event_observed_B=not_admited['DECEDES'])
    print(results.summary)

if __name__ == '__main__':
    run()

