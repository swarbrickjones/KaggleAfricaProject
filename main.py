import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error
from  sklearn import linear_model 

from sklearn import ensemble

from sklearn import linear_model
from sklearn import cross_validation
from os import path
import math

#resource.setrlimit(resource.RLIMIT_AS, (10000 * 1048576L, -1L))

    ### data handling ###       
print('reading files')


train_data_raw = pd.read_csv(path.relpath("data/training.csv"))   ## pandas data frame

target_features = ['Ca','P','pH','SOC','Sand']
y_features = train_data_raw[target_features]
train_data_raw.drop(target_features,axis=1, inplace=True)

test_data_raw = pd.read_csv(path.relpath("data/test.csv"))     ## pandas data frame

def split_data(train_data_raw):
    ids = np.array(train_data_raw['PIDN'])
    train_data_raw.drop('PIDN',1, inplace=True)
    spacial_features = ['BSAN','BSAS','BSAV',  \
                        'CTI', 'ELEV', 'EVI', 'LSTD','LSTN',  \
                        'REF1','REF2','REF3','REF7','RELI', \
                        'TMAP', 'TMFI','Depth']
    X_spacial = train_data_raw[spacial_features].as_matrix()
    
    train_data_raw.drop(spacial_features,axis=1, inplace=True)
    X_ir = train_data_raw.as_matrix()
    return ids, X_spacial, X_ir
    
train_ids, train_X_spacial, train_X_ir = split_data(train_data_raw)
test_ids, test_X_spacial, test_X_ir = split_data(test_data_raw)

return_matrix = pd.DataFrame(data = test_ids, columns = ['PIDN'])

for feature_name in y_features.columns.values:
    print 'training for feature : ', feature_name
    y = np.array(y_features[feature_name])
    cv = cross_validation.StratifiedKFold(range(len(y)), n_folds = 10)
    c_best = 0
    c_best_score = 10000.0
    for C in [10 * i for i in range(5,10)]:
        print 'C = ',  C,
        #clf = linear_model.Ridge(alpha = C)  
        clf = ensemble.RandomForestRegressor(n_estimators = C)
        mse_list = []
        for i, (train, val) in enumerate(cv):
            print 'Fold ', i,
            X_train, X_val = train_X_ir[train], train_X_ir[val]
            y_train, y_val = y[train], y[val]
            
            
            y_pred = clf.fit(X_train,y_train).predict(X_val)
            mse = mean_squared_error(y_pred, y_val)
            mse_list.append(mse)
            print ', mse = ', mse
        mean_mse = sum(mse_list) / len(mse_list)
        print ', mean mse = ' , mean_mse
        if(mean_mse < c_best_score):
            c_best_score= mean_mse
            c_best = C 
            
    print 'best C = ', c_best
    #clf = linear_model.Ridge(alpha = c_best)
    clf = ensemble.RandomForestRegressor(n_estimators = c_best)
    y_pred = clf.fit(train_X_ir, y).predict(test_X_ir)
    return_matrix[feature_name] = y_pred
    print return_matrix

return_matrix.to_csv(path.relpath("data/output2.csv"), index = False)
    





     
    ###  train classifiers  ###
    
    #nlp classifier   
    
#    print 'getting nlp scores'
#    
#    y_train = np.asarray(train_data_raw["requester_received_pizza"],dtype = 'bool')  
#    
#    #nlp_clf_title = RawDataClassifier(NLPClassifier (), NLPEngineer('request_title', max_features_ = 1000))
#    #nlp_clf_title.fit(train_data_raw, y_train)    
#    
#    #metadata classifier
#    
#    print 'getting meta data scores'
#    
#    meta_clf1 = ensemble.GradientBoostingClassifier(n_estimators = 40)
#    meta_clf12 = ensemble.GradientBoostingClassifier(n_estimators = 10)    
#    meta_clf2 = ensemble.RandomForestClassifier(n_estimators = 300, n_jobs = -1)
#    meta_clf3 = linear_model.LogisticRegression(C = 1) 
#    nlp_clf = linear_model.LogisticRegression(C = 0.4) #
#    nlp_clf2 =  linear_model.LogisticRegression(C = 50) # NLPClassifier ()
#    estimators = [meta_clf1,meta_clf12, meta_clf2, meta_clf3, nlp_clf, nlp_clf2]
# 
#    meta_engineer = MetadataEngineer()
#    X_meta_train = meta_engineer.transform(train_data_raw)
#
#    date_engineer = DateEngineer()
#    X_date_train = date_engineer.transform(train_data_raw)
#    
#    nlp_engineer = NLPEngineer('request_text_edit_aware', max_features_ = 1000000)
#    X_nlp_train = nlp_engineer.transform(train_data_raw)
#    
#    nlp_engineer2 = NLPEngineer('request_title', max_features_ = 1000000)
#    X_nlp_train2 = nlp_engineer2.transform(train_data_raw)
#    
#    input_train = [X_meta_train,X_date_train,X_meta_train,X_meta_train,X_nlp_train,X_nlp_train2]
#    
#    stacking = Stacking(LogisticRegression(C=1), estimators,
#                 folds = 10, raw = True
#                 )
#    
#    stacking.fit(input_train, y_train)
#    
#    X_meta_test = meta_engineer.transform(test_data_raw)  
#    X_date_test = date_engineer.transform(test_data_raw)  
#    X_nlp_test = nlp_engineer.transform(test_data_raw)
#    X_nlp_test2 = nlp_engineer2.transform(test_data_raw)    
#    input_test = [X_meta_test,X_date_test,X_meta_test,X_meta_test,X_nlp_test,X_nlp_test2]    
#            
#    y_test_pred = stacking.predict_proba(input_test)[:, 1]
#    
#    test_ids=test_data_raw['request_id']    
#
#    print 'writing to file'    
#    
#    fcsv = open('raop_prediction.csv','w')
#    fcsv.write("request_id,requester_received_pizza\n")
#    for index in range(len(y_test_pred)):
#        theline = str(test_ids[index]) + ',' + str(y_test_pred[index])+'\n'
#        fcsv.write(theline)
#    
#    fcsv.close()
#    
    ###   word bag scoring   #####
   
    
    
    

##if __name__ == '__main__':
##    run()