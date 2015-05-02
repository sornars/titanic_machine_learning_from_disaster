import pandas as pd
import numpy as np
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report

def munge_data(csv_input):
    """Take train and test set and make them useable for machine learning algorithms."""
    df = pd.read_csv(csv_input)
    
    df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

    most_frequent_port = df['Embarked'].describe().top
    df['EmbarkedMap'] = df['Embarked'].fillna(most_frequent_port)
    df['EmbarkedMap'] = map_strings_to_categories(df['EmbarkedMap'])

    # df['Honorific'] = df['Name'].str.lower().str.extract('([a-z]+\.)')
    # df['Honorific'] = map_strings_to_categories(df['Honorific'])

    median_age = df['Age'].dropna().median()
    df['AgeFill'] = df['Age'].fillna(median_age)

    median_fare_by_class = df.groupby(['Pclass'])['Fare'].median()
    classes = df['Pclass'].unique()
    for pclass in classes:
        df.loc[(df['Fare'].isnull()) & (df['Pclass'] == pclass), 'Fare'] = median_fare_by_class[pclass]

    df['Child'] = (df['Age'] < 18).astype(int)
    df['Age*Class'] = df['AgeFill'] * df['Pclass']
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['TicketPrefix'] = df['Ticket'].str.split(' ').str[0].replace('^[0-9]+$', '0', regex=True)
    df['TicketPrefix'] = map_strings_to_categories(df['TicketPrefix'])

    df = df.drop(['Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)

    return df

def map_strings_to_categories(s):
    """Extract unique values from a Series and return the series mapped to numbers."""
    uniques = s.unique()
    uniques.sort()
    s_dict = {value: i for i, value in enumerate(uniques)}
    return s.map(s_dict).astype(int)

def main():
    train_df = munge_data('./data/train.csv')
    train_df = train_df.drop('PassengerId', axis=1)

    cx_df = train_df[800:]
    cx_target_df = cx_df['Survived']
    cx_df = cx_df.drop('Survived', axis=1)
    cx_df = cx_df.sort(axis=1)
    
    train_df = train_df[:800]
    target_df = train_df['Survived']
    train_df = train_df.drop('Survived', axis=1)
    train_df = train_df.sort(axis=1)

    test_df = munge_data('./data/test.csv')
    test_ids = test_df.PassengerId.values
    test_df = test_df.drop('PassengerId', axis=1)
    test_df = test_df.sort(axis=1)
    
    cx_data = cx_df.values
    cx_target_data = cx_target_df.values
    train_data = train_df.values
    target_data = target_df.values
    test_data = test_df.values

    gnb = GaussianNB()
    gnb.fit(train_data, target_data)
    cx_predictions = gnb.predict(cx_data)

    print(classification_report(cx_target_data, cx_predictions))

    predictions = gnb.predict(test_data)

    with open('output.csv', 'w') as o:
        o.write('PassengerId,Survived\n')
        for passenger, prediction in zip(test_ids, predictions):
            o.write('{},{}\n'.format(passenger, prediction))

if __name__ == '__main__':
    main()
