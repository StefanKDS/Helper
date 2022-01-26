########################################################
# Dense Model ( Classification )
########################################################
import tensorflow as tf
from sklearn.compose import make_column_transformer
from keras.callbacks import ReduceLROnPlateau, EarlyStopping


# Callbacks
reduce_lr = ReduceLROnPlateau(monitor='val_accuracy', min_lr=0.0001, patience=5, verbose=1)
early_stopping = EarlyStopping(patience=20, monitor='val_accuracy')
callbacks = [early_stopping,reduce_lr]

# Set random seed
tf.random.set_seed(42)

# Build the model (2 layers, 250, 1 units)
model_1 = tf.keras.Sequential([
  tf.keras.layers.Dense(100, activation="relu"),
  tf.keras.layers.Dense(250, activation="relu"),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile the model
model_1.compile(loss = 'binary_crossentropy',
                          optimizer=tf.keras.optimizers.Adam(learning_rate=0.2),
                          metrics=['accuracy'])

# Fit the model
history = model_1.fit(X_train, y_train, epochs=100, callbacks=callbacks, validation_data=(X_test, y_test))


########################################################
# Random Forest ( Classification )
########################################################
# Train the model
from sklearn.ensemble import RandomForestClassifier

model2 = RandomForestClassifier(max_depth=100, random_state=1)
model2.fit(X_train, y_train)

model2.score(X_test, y_test)


########################################################
# Logistic Regression ( Classification )
########################################################
from sklearn.linear_model import LogisticRegression
model3 = LogisticRegression(random_state=0)
model3.fit(X_train, y_train)
predictions = model3.predict(X_test)
score = model3.score(X_test, y_test)
print(score)