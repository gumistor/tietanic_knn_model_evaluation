import pandas as pd
df = pd.read_csv(r"data\train.csv")


#C_1 = df[["Name","Pclass"]][4:9]
#C_1 = df.loc[:,["Name","Pclass"]]
C_1 = df.Name[0:9]

print(C_1)