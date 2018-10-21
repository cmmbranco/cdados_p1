import pandas as pd
import numpy as np
 
# 
# 
# 
# 
# 
# ####
# ####Main Loop
# ####
# 
# 
# 
# data = pd.read_csv("../scania_dataset/aps_failure_training_set.csv")
# 
# 
# featureNames = data.columns.values
# 
# classes = data['class']
# nclasses = pd.unique(classes)
# #print(classes)
# 
# 
# totalentries = len(classes)
# print(totalentries)
# 
# dic = {}
# 
# 
# ##calculate % of na per feature
# for column in data:
#     dic[column] = 0
#     
#     for x in data[column]:
#         if x == 'na':
#             dic[column] += 1
#     
# 
# for word in dic:
#     dic[word] = (dic[word]/totalentries)*100.0
# 
# for x in dic:
#     if dic[x] >= 60:
#         print(f"{x} with {dic[x]}% missing")
#         
# 
# 
# 
# ##extract positive entries for resampling
# posframe = pd.DataFrame(columns=(featureNames))
# counter = 0
# pos = 0
# while counter < totalentries:
#     row = data.iloc[counter]
#     if "pos" in row[0]:
#         posframe.loc[pos] = row
#         pos += 1
#          
#     counter += 1
# 
# 
# ##detect features with na
# 
# counter = 0
# postlen = len(posframe['class'])
# 
# while counter < postlen:
#     print(posframe.iloc[counter])
#     counter += 1
# 
# 
# print(posframe)
#         
# 
# # sorted_by_value = sorted(dic.items(), key=lambda kv: kv[1])
# # 
# # #print(sorted_by_value)
# # 
# # for entry in sorted_by_value:
# #     print(entry)
# 
# 
#     
# # for word in dic:
# #     print(f"line {word} with {dic[word]} na")
#     
# 
# 
# 
# 
# 
# # while counter < totalentries:
# #     dic[counter] = 0
# #     bla = data.iloc[counter]
# #     #print(bla)
# #     for x in bla:
# #         if x == 'na':
# #             dic[counter] += 1
# #          
# #     counter += 1
#     
# 
# 
# #print(sort)
