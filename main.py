'''
Created on 1 gru 2020

@author: gumistorro
'''

import time
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn import preprocessing

start_time = time.perf_counter()
print("{:15.7f}".format(start_time),"-"*4,"Start","-"*10)
#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
pd.set_option('display.max_columns', None)
pd.set_option('display.expand_frame_repr', False)

trainData_raw = pd.read_csv(r"C:/Users/obi1i/OneDrive/Matlab/ML-Titanic/data/train.csv")
testVectors_raw = pd.read_csv(r"C:/Users/obi1i/OneDrive/Matlab/ML-Titanic/data/test.csv")
testVectors_ans = pd.read_csv(r"C:/Users/obi1i/OneDrive/Matlab/ML-Titanic/data/test_ans.csv",sep = ';')
#names = ['x','y','z'], usecols = range(3), encoding = " " - some possible parameers, that can shape 
#                                                            the content of DataFrame

#Survided processing
trainData = trainData_raw
xxx = pd.Categorical(trainData['Survived'], categories=[0, 1, 2], ordered = False).rename_categories({0:"Died",1:"Survived",2:"N/A"})
trainData['Survived'] = xxx

testVectors = testVectors_raw
testVectors['Survived'] = pd.Categorical(["N/A"]*testVectors.shape[0], categories=["Died", "Survived", "N/A"], ordered = False)
testVectors = testVectors.reindex(columns = (trainData.columns))

#Concatenate train and test data to create whole data set with a categorical argument to distinguh them
trainData['TestVector'] = [0]*trainData.shape[0]
testVectors['TestVector'] = [1]*testVectors.shape[0]
 
allData = pd.concat([trainData, testVectors])

#Pclass processing
allData['Pclass'] = allData['Pclass'].astype('category').cat.rename_categories({1:"FirstClass", 2:"SecondClass", 3:"ThirdClass"})

#Name processing
temp = allData['Name'].str.split(",",expand=True)
allData['Name'] = temp[0].str.strip()
temp = temp[1].str.split(".",expand=True)
allData['Title'] = temp[0].str.strip()
 
allData['Title'] = allData['Title'].astype('category')
 
allData = allData.reindex(columns = (list(allData.columns[0:3])+[allData.columns[-1]]+list(allData.columns[3:-1])))
 
#allData[allData['Title'] == 'Capt'] = 'Mr' 

allData['Title'] = allData['Title'].cat.add_categories('Noble male')

allData.loc[allData["Title"] == "Capt", "Title"] = "Noble male"
allData.loc[allData["Title"] == "Col", "Title"] = "Noble male"
allData.loc[allData["Title"] == "Major", "Title"] = "Noble male"
allData.loc[allData["Title"] == "Don", "Title"] = "Noble male"
allData.loc[allData["Title"] == "Sir", "Title"] = "Noble male"
allData.loc[allData["Title"] == "Jonkheer", "Title"] = "Noble male"
allData.loc[allData["Title"] == "Rev", "Title"] = "Noble male"

allData.loc[allData["Title"] == "Ms", "Title"] = "Mrs"
allData.loc[allData["Title"] == "Miss", "Title"] = "Mrs"
allData.loc[allData["Title"] == "Mlle", "Title"] = "Mrs"
allData.loc[allData["Title"] == "Dona", "Title"] = "Mrs"
allData.loc[allData["Title"] == "Mme", "Title"] = "Mrs"
allData.loc[allData["Title"] == "the Countess", "Title"] = "Mrs"
allData.loc[allData["Title"] == "Lady", "Title"] = "Mrs"

allData["Title"] = allData["Title"].cat.remove_unused_categories()

#Sex processing
allData['Sex'] = allData['Sex'].astype('category')

#Age processing
males_firstClass_over16 = (allData.Sex == 'male') & (allData.Pclass == 'FirstClass') & (allData.Age > 16)
males_secondClass_over16 = (allData.Sex == 'male') & (allData.Pclass == 'SecondClass') & (allData.Age > 16)
males_thirdClass_over16 = (allData.Sex == 'male') & (allData.Pclass == 'ThirdClass') & (allData.Age > 16)


male_firstclass_mean = allData.loc[males_firstClass_over16,"Age"].mean()
male_secondclass_mean = allData.loc[males_secondClass_over16,"Age"].mean()
male_thirdclass_mean = allData.loc[males_thirdClass_over16,"Age"].mean()


male_firstclass_median = allData.loc[males_firstClass_over16,"Age"].median()
male_secondclass_median = allData.loc[males_secondClass_over16,"Age"].median()
male_thirdclass_median = allData.loc[males_thirdClass_over16,"Age"].median()


male_firstclass_count = allData.loc[males_firstClass_over16,"Age"].shape[0]
male_secondclass_count = allData.loc[males_secondClass_over16,"Age"].shape[0]
male_thirdclass_count = allData.loc[males_thirdClass_over16,"Age"].shape[0]


males_missingAge = (allData.Sex == 'male') & (allData.Age.isnull());
males_missignAge_noRelatives = males_missingAge & (allData.SibSp == 0) & (allData.Parch == 0);

males_missingAge_1st_noRelatives = (allData.Pclass == 'FirstClass') & (males_missignAge_noRelatives)
males_missingAge_2nd_noRelatives = (allData.Pclass == 'SecondClass') & (males_missignAge_noRelatives)
males_missingAge_3rd_noRelatives = (allData.Pclass == 'ThirdClass') & (males_missignAge_noRelatives)

allData.loc[males_missingAge_1st_noRelatives,"Age"] = male_firstclass_median
allData.loc[males_missingAge_2nd_noRelatives,"Age"] = male_secondclass_median
allData.loc[males_missingAge_3rd_noRelatives,"Age"] = male_thirdclass_median

#female
females_firstClass_over16 = (allData.Sex == 'female') & (allData.Pclass == 'FirstClass') & (allData.Age > 16)
females_secondClass_over16 = (allData.Sex == 'female') & (allData.Pclass == 'SecondClass') & (allData.Age > 16)
females_thirdClass_over16 = (allData.Sex == 'female') & (allData.Pclass == 'ThirdClass') & (allData.Age > 16)

female_firstclass_mean = allData.loc[females_firstClass_over16,"Age"].mean()
female_secondclass_mean = allData.loc[females_secondClass_over16,"Age"].mean()
female_thirdclass_mean = allData.loc[females_thirdClass_over16,"Age"].mean()

female_firstclass_median = allData.loc[females_firstClass_over16,"Age"].median()
female_secondclass_median = allData.loc[females_secondClass_over16,"Age"].median()
female_thirdclass_median = allData.loc[females_thirdClass_over16,"Age"].median()

female_firstclass_count = allData.loc[females_firstClass_over16,"Age"].shape[0]
female_secondclass_count = allData.loc[females_secondClass_over16,"Age"].shape[0]
female_thirdclass_count = allData.loc[females_thirdClass_over16,"Age"].shape[0]

females_missingAge = (allData.Sex == 'female') & (allData.Age.isnull())
females_missignAge_noRelatives = (females_missingAge) & (allData.SibSp == 0) & (allData.Parch == 0)

females_missingAge_1st_noRelatives = (allData.Pclass == 'FirstClass') & (females_missignAge_noRelatives)
females_missingAge_2nd_noRelatives = (allData.Pclass == 'SecondClass') & (females_missignAge_noRelatives)
females_missingAge_3rd_noRelatives = (allData.Pclass == 'ThirdClass') & (females_missignAge_noRelatives)

allData.loc[females_missingAge_1st_noRelatives,"Age"] = female_firstclass_median;
allData.loc[females_missingAge_2nd_noRelatives,"Age"] = female_secondclass_median;
allData.loc[females_missingAge_3rd_noRelatives,"Age"] = female_thirdclass_median;

#young male

males_firstClass_under18 = (allData.Sex == 'male') & (allData.Pclass == 'FirstClass') & (allData.Age < 18)
males_secondClass_under18 = (allData.Sex == 'male') & (allData.Pclass == 'SecondClass') & (allData.Age < 18)
males_thirdClass_under18 = (allData.Sex == 'male') & (allData.Pclass == 'ThirdClass') & (allData.Age < 18)

youngmale_firstclass_mean = allData.loc[males_firstClass_under18,"Age"].mean()
youngmale_secondclass_mean = allData.loc[males_secondClass_under18,"Age"].mean()
youngmale_thirdclass_mean = allData.loc[males_thirdClass_under18,"Age"].mean()

youngmale_firstclass_median = allData.loc[males_firstClass_under18,"Age"].median()
youngmale_secondclass_median = allData.loc[males_secondClass_under18,"Age"].median()
youngmale_thirdclass_median = allData.loc[males_thirdClass_under18,"Age"].median()
 
youngmale_firstclass_count = allData.loc[males_firstClass_under18,"Age"].shape[0]
youngmale_secondclass_count = allData.loc[males_secondClass_under18,"Age"].shape[0]
youngmale_thirdclass_count = allData.loc[males_thirdClass_under18,"Age"].shape[0]

males_missingAge = ((allData.Title == 'Master') | ((allData.Parch > 0) & (allData.SibSp > 0))) & ((allData.Age.isnull()));

males_missingAge_1st = (allData.Pclass == 'FirstClass') & (males_missingAge)
males_missingAge_2nd = (allData.Pclass == 'SecondClass') & (males_missingAge)
males_missingAge_3rd = (allData.Pclass == 'ThirdClass') & (males_missingAge)

allData.loc[males_missingAge_1st,"Age"] = youngmale_firstclass_median
allData.loc[males_missingAge_2nd,"Age"] = youngmale_secondclass_median
allData.loc[males_missingAge_3rd,"Age"] = youngmale_thirdclass_median

all_missingAge = (allData.Age.isnull()) & (allData.SibSp > 0)

sameNames = allData.loc[all_missingAge]
sameNames_size = sameNames.shape[0]
for idx in range(0,sameNames_size):
    idx_name_value = sameNames.iloc[idx].Name
    idx_record_value = sameNames.iloc[idx].PassengerId
    idx_ticket_value = sameNames.iloc[idx].Ticket
    other = allData.loc[(allData.Ticket == idx_ticket_value) & (~allData.Age.isnull())]
    if other.shape[0] > 0:
        x = (other.Age.max());
        allData.loc[(allData.PassengerId == idx_record_value),"Age"] = x+2;
        
allData.loc[(allData.Age.isnull()) & (allData.Sex == "male") & (allData.Pclass == "FirstClass") ,'Age'] = male_firstclass_median
allData.loc[(allData.Age.isnull()) & (allData.Sex == "female") & (allData.Pclass == "FirstClass") ,'Age'] = female_firstclass_median
allData.loc[(allData.Age.isnull()) & (allData.Sex == "male") & (allData.Pclass == "SecondClass") ,'Age'] = male_secondclass_median
allData.loc[(allData.Age.isnull()) & (allData.Sex == "female") & (allData.Pclass == "SecondClass") ,'Age'] = female_secondclass_median
allData.loc[(allData.Age.isnull()) & (allData.Sex == "male") & (allData.Pclass == "ThirdClass") ,'Age'] = male_thirdclass_median
allData.loc[(allData.Age.isnull()) & (allData.Sex == "female") & (allData.Pclass == "ThirdClass") ,'Age'] = female_thirdclass_median

allData.Age = pd.cut(allData.Age, [0, 12, 18, 24, 30, 50, 100] ,labels=["Child","Teenage","YoungAdult","Adult","Mature","Old"])

#Fare processing
allData['Embarked'] = allData['Embarked'].astype('category')

allData['PassNo'] = allData.SibSp+allData.Parch+1;
allData['Fare_new'] = allData.Fare/allData.PassNo;

#SibSp and Psrch processing

for index, row in allData.iterrows():
    if row.PassNo > 1:
        if len(row.Ticket)>0:
            sex_count = allData.loc[allData.Ticket == row.Ticket].Sex.value_counts() #distribution of data, number of unique values
        else:
            sex_count = allData.loc[allData.Name == row.Name].Sex.value_counts()
            
        male_relatives = sex_count['male'] 
        female_relatives = sex_count['female']
        
        allData.loc[index,'Female_Relatives'] = male_relatives+female_relatives
        #print("Ticket:",male_relatives,female_relatives)

allData.loc[allData.Female_Relatives.isnull(),'Female_Relatives'] = 0

allData['Female_Relatives'] = pd.cut(allData.Female_Relatives+1, [0, 1, 2, 4, 10] ,labels=['Alone','Spouse','SmallFamily','BigFamily'])

#cabin processing
allData.loc[allData.Cabin.isnull(),'Cabin'] ='Z'
allData['Deck'] = allData['Cabin'].astype(str).str[0]
allData['Deck'] = allData.Deck.astype('category')

#print(allData)


allData.Fare = allData.Fare_new 
allData = allData.drop(columns=['PassNo','Fare_new','Sex','Embarked','Fare','Ticket','SibSp','Parch','Cabin','Deck','Name'])

cat_columns = allData.select_dtypes(['category']).columns
allData[cat_columns] = allData[cat_columns].apply(lambda x: x.cat.codes)

trainDataP = allData[allData.TestVector == 0]
testDataP = allData[allData.TestVector == 1]

trainDataP_ans = trainDataP.Survived
trainDataP = trainDataP.drop(columns=['Survived','TestVector','PassengerId'])

resultData = pd.DataFrame(testDataP.PassengerId)
testDataP = testDataP.drop(columns=['Survived','TestVector','PassengerId'])

knn = KNeighborsClassifier(n_neighbors=11,weights='uniform',algorithm='brute', leaf_size=20, p=2, metric='minkowski',
                                       metric_params=None,n_jobs=None)

knn.fit(trainDataP,trainDataP_ans)

resultData['Survived'] = knn.predict(testDataP)

file_header = 666

resultData.to_csv(str("result_{}.csv".format(file_header)),sep=",",index=False)

print(resultData)
#at the end remove some doata

    #end
#print(males_missingAge_1st_noRelatives,males_missingAge_2nd_noRelatives,males_missingAge_3rd_noRelatives)

#print(male_firstclass_median,male_secondclass_median,male_thirdclass_median)
#print(male_firstclass_count,male_secondclass_count,male_thirdclass_count)

#---------------------------------------------------------------------------
#---------------------------------------------------------------------------
print("{:15.7f}".format(time.perf_counter()-start_time),"-"*4,"Stop","-"*11)