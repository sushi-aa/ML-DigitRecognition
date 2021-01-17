#creidt to chribsen on github

from sklearn.linear_model import LogisticRegression

model = LogisticRegression()

#[height, weight, shoe_size]
X = [[181, 80, 44], [177, 70, 43], [160, 60, 38], [154, 54, 37], [166, 65, 40], [190, 90, 47], [175, 64, 39],
     [177, 70, 40], [159, 55, 37], [171, 75, 42], [181, 85, 43]]

Y = ['male', 'male', 'female', 'female', 'male', 'male', 'female', 'female', 'female', 'male', 'male']

model.fit(X, Y)

prediction = model.predict([[190, 70, 43]])
prediction2 = model.predict([[188, 60, 50]])
prediction3 = model.predict([[180, 80, 45]])

print (prediction)
print(prediction2)
print(prediction3)