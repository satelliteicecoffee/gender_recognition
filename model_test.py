## Gender recognition test module
## Modify to suit motor sound inspection
## https://blog.csdn.net/leilei7407/article/details/103856584
## https://blog.csdn.net/peinbill/article/details/80106790

##  Preprocessing ##
# import train data 
import pandas as pd
import joblib
voice_data = pd.read_csv('test_set/train_short.csv')
x = voice_data.iloc[:,:-1]
y = voice_data.iloc[:,-1]

# import test data
# test_data = pd.read_csv('test_set/ow_test_fake.csv')
# xt = test_data.iloc[:,:-1]
# yt = test_data.iloc[:,-1]

print(y)

# transform label
from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y = le.fit_transform(y)

# feature missing value completion
from sklearn.impute import SimpleImputer
imp = SimpleImputer(missing_values=0, strategy='mean')
x = imp.fit_transform(x)

# use same normalization method on test set as on train set
# from sklearn.utils import shuffle
# xt = imp.transform(xt)
# yt = le.transform(yt)
# xt, yt = shuffle(xt, yt)

# order random
# x_train: train set extruded feature array/matrix
# y_train: train set label
# x_test: test set extruded feature array/matrix
# y_test: test set label
from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.15)

# Normalization
from sklearn.preprocessing import StandardScaler
scaler1 = StandardScaler()
scaler1.fit(x_train)
joblib.dump(scaler1, f'model_set/scaler1.joblib')
x_train = scaler1.transform(x_train)
x_test = scaler1.transform(x_test)


## Modelling ##
# Logistic
from sklearn.linear_model import LogisticRegression
logistic = LogisticRegression(max_iter=10000)
logistic.fit(x_train, y_train) # fit model with given train set
joblib.dump(logistic, f'model_set/logistic.joblib')

# Neural network
from sklearn.neural_network import MLPClassifier
nn = MLPClassifier(max_iter=100000)
nn.fit(x_train, y_train)
joblib.dump(nn, f'model_set/nn.joblib')

# Cart
from sklearn.tree import DecisionTreeClassifier
cart = DecisionTreeClassifier()
cart.fit(x_train, y_train)
joblib.dump(cart, f'model_set/cart.joblib')

# Random forest
from sklearn.ensemble import RandomForestClassifier
rf = RandomForestClassifier(n_estimators=10, criterion="gini")
rf.fit(x_train, y_train)
joblib.dump(rf, f'model_set/rf.joblib')

# SVM
from sklearn.svm import SVC
svc = SVC(C=1, kernel='rbf', probability=True)
svc.fit(x_train, y_train)
joblib.dump(svc, f'model_set/svc.joblib')

# KNN
from sklearn.neighbors import KNeighborsClassifier
knn = KNeighborsClassifier()
knn.fit(x_train, y_train)
joblib.dump(knn, f'model_set/knn.joblib')


## Validation, Accuracy Assessment ##
from sklearn import metrics
train_methods = [logistic, nn, cart, rf, svc, knn]
for train_method in train_methods:
    y_train_result = train_method.predict(x_train)  # predict x_train results
    print(f'{train_method} train score:')
    print(metrics.accuracy_score(y_train_result, y_train))
    y_pred = train_method.predict(x_test)
    print(f'{train_method} test score:')
    # print(train_method.predict_proba(x_test))
    print(metrics.accuracy_score(y_test, y_pred))  # correct input order
    print('\n')
    # yt_predic = train_method.predict(xt)
    # print(f'{train_method} isolate test set score:')
    # print(metrics.accuracy_score(yt, yt_predic))
    # print('\n')
    # for item in nn.predict_proba(xt):
    #     print(item)



pass

# Next step: 
# Get sound sample, modify to extract 20 sound parameter
# Build constant loop, capture from mic and output prediction results           
