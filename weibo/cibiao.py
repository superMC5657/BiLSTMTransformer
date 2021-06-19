import pickle
file=open("weibo/data/vocab.pkl","rb")
data=pickle.load(file)
print(data)
file.close()