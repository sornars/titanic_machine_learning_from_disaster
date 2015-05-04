import pandas as pd
import numpy as np
from sklearn import cross_validation, svm
from sklearn.metrics import classification_report

gender_map =  {'female': 0, 'male': 1}
embarked_map = {'C': 0, 'S': 1, 'Q': 2}
title_map = {'capt.': 0, 'col.': 1, 'countess.': 2, 'don.': 3, 'dr.': 4,
             'jonkheer.': 5, 'lady.': 6, 'major.': 7, 'master.': 8, 
             'miss.': 9, 'mlle.': 10, 'mme.': 11, 'mr.': 12, 'mrs.': 13,
             'ms.': 14, 'rev.': 15, 'sir.': 16, 'dona.': 17}
pclasses = [1, 2, 3]
ticket_map = {'0': 3, 'A': 0, 'C': 4, 'F': 6, 'L': 7, 'P': 1, 'S': 2, 'W': 5}
cabin_map = {'A': 5, 'B': 6, 'C': 1, 'D': 4, 'E': 2, 'F': 7, 'G': 3, 'T': 8, 'U': 0}
median_age = None
median_fare_by_class = pd.Series()

def munge_data(csv_input):
    """Take train and test set and make them useable for machine learning algorithms."""
    global median_age, median_fare_by_class
    df = pd.read_csv(csv_input)
    
    df['Sex'] = df['Sex'].map(gender_map)

    df['Embarked'] = df['Embarked'].fillna('S').map(embarked_map)

    df['Title'] = df['Name'].str.lower().str.extract('([a-z]+\.)').map(title_map)

    if not median_age:
        median_age = df['Age'].dropna().median()
    df['Age'] = df['Age'].fillna(median_age)

    if median_fare_by_class.empty:
        median_fare_by_class = df.groupby(['Pclass'])['Fare'].median()
    for pclass in pclasses:
        df.loc[(df['Fare'].isnull()) & (df['Pclass'] == pclass), 'Fare'] = median_fare_by_class[pclass]

    df['Child'] = (df['Age'] < 18).astype(int)
    df['Age*Class'] = df['Age'] * df['Pclass']
    df['FamilySize'] = df['SibSp'] + df['Parch']
    df['Ticket'] = df['Ticket'].str.split(' ').str[0].replace('^[0-9]+$', '0', regex=True).str[0].map(ticket_map)
    
    df = df.drop(['Name', 'Cabin'], axis=1)
    
    return df

def main():
    train_df = munge_data('./data/train.csv')
    train_df = train_df.drop('PassengerId', axis=1)
    target_df = train_df['Survived']
    train_df = train_df.drop('Survived', axis=1)
    train_df = train_df.sort(axis=1)

    test_df = munge_data('./data/test.csv')
    test_ids = test_df.PassengerId.values
    test_df = test_df.drop('PassengerId', axis=1)
    test_df = test_df.sort(axis=1)
    
    train_data = train_df.values
    target_data = target_df.values
    test_data = test_df.values

    scores = []
    for i in range(0, 5):
        train_data, cx_data, target_data, cx_target_data = cross_validation.train_test_split(
            train_data, target_data, test_size=0.2)

        clf = svm.SVC(kernel='linear')
        clf.fit(train_data, target_data)
        scores.append(clf.score(cx_data, cx_target_data))
        cx_predictions = clf.predict(cx_data)
        print(classification_report(cx_target_data, cx_predictions))

    avg_score = sum(scores)/len(scores)
    print(avg_score)

    predictions = clf.predict(test_data)

    with open('output.csv', 'w') as o:
        o.write('PassengerId,Survived\n')
        for passenger, prediction in zip(test_ids, predictions):
            o.write('{},{}\n'.format(passenger, prediction))

if __name__ == '__main__':
    main()
