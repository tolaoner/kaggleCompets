import pandas as pd
import tensorflow as tf
from tensorflow.keras import layers
pd.options.display.max_columns = 15
train_data = pd.read_csv('train_ready.csv')
# print(train_data)
# define feature columns
feature_columns = []
passengerid = tf.feature_column.numeric_column('PassengerId')
feature_columns.append(passengerid)

pclass = tf.feature_column.numeric_column('Pclass')
feature_columns.append(pclass)

# sex = tf.feature_column.numeric_column('Sex')
categ_sex = tf.feature_column.categorical_column_with_vocabulary_list('Sex', ['Male', ' Female'])
sex = tf.feature_column.indicator_column(categ_sex)
feature_columns.append(sex)

age = tf.feature_column.numeric_column('Age')
feature_columns.append(age)

sibsp = tf.feature_column.numeric_column('SibSp')
feature_columns.append(sibsp)

parch = tf.feature_column.numeric_column('Parch')
feature_columns.append(parch)

fare = tf.feature_column.numeric_column('Fare')
feature_columns.append(fare)

# embarked = tf.feature_column.numeric_column('Embarked')
categ_embark = tf.feature_column.categorical_column_with_vocabulary_list('Embarked', ['S', 'C', 'Q'])
embarked = tf.feature_column.indicator_column(categ_embark)
feature_columns.append(embarked)
# print(feature_columns)

my_feature_layer = layers.DenseFeatures(feature_columns)


def create_deep_model(learning_rate, feature_layer):
    model = tf.keras.models.Sequential()
    model.add(feature_layer)
    model.add(layers.Dense(units=20,
                           activation="relu",
                           name='hidden1'))
    model.add(layers.Dense(units=12,
                           activation="relu",
                           name="hidden2"))
    model.add(layers.Dense(units=1,
                           name="output"))
    model.compile(optimizer=tf.keras.optimizers.Adam(lr=learning_rate),
                  metrics=[tf.keras.metrics.Accuracy()])
    return model


def train_model(model, dataset, epochs, batch_size, label_name):
    features = dataset.drop(label_name, axis='columns')
    label = dataset[label_name]
    history = model.fit(x=features,
                        y=label,
                        batch_size=batch_size,
                        epochs=epochs,
                        shuffle=True)
    iterations = history.epoch
    hist = pd.DataFrame(history.history)
    accuracy = hist['accuracy']
    return iterations, accuracy


'''my_model = create_deep_model(0.1, my_feature_layer)
eps, acc = train_model(my_model, train_data, 200, 25, 'Survived')
'''