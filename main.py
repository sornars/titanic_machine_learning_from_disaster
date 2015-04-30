import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier 

def munge_data(csv_input):
	"""Take train and test set and make them useable for machine learning algorithms."""
	df = pd.read_csv(csv_input)
	
	df['Gender'] = df['Sex'].map({'female': 0, 'male': 1}).astype(int)

	most_frequent_port = df['Embarked'].describe().top
	df['EmbarkedMap'] = df['Embarked'].fillna(most_frequent_port)
	port_cities = df['EmbarkedMap'].unique()
	port_cities.sort()
	port_dict = {port: i for i, port in enumerate(port_cities)}
	df['EmbarkedMap'] = df['EmbarkedMap'].map(port_dict).astype(int)

	median_age = df['Age'].dropna().median()
	df['AgeFill'] = df['Age'].fillna(median_age)

	median_fare_by_class = df.groupby(['Pclass'])['Fare'].median()
	classes = df['Pclass'].unique()
	classes.sort()
	for pclass in classes:
		df.loc[(df['Fare'].isnull()) & (df['Pclass'] == pclass), 'Fare'] = median_fare_by_class[pclass]

	df = df.drop(['Age', 'Name', 'Sex', 'Ticket', 'Cabin', 'Embarked'], axis=1)
	df.sort()

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

	forest = RandomForestClassifier(n_estimators = 100)
	forest = forest.fit(train_data, target_data)
	output = forest.predict(test_data)
	
	with open('output.csv', 'w') as o:
		o.write('PassengerId,Survived\n')
		for passenger, prediction in zip(test_ids, output):
			o.write('{},{}\n'.format(passenger, prediction))

if __name__ == '__main__':
	main()
