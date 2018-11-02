import pandas as pd
import numpy as np

green_data = pd.read_csv('../col_dataset/green.csv')
hinselmann_data = pd.read_csv('../col_dataset/hinselmann.csv')
schiller_data = pd.read_csv('../col_dataset/schiller.csv')

green_attributes = green_data.columns.values
hinselmann_attributes = hinselmann_data.columns.values
schiller_attributes = schiller_data.columns.values

green_outlier_info = {}
hinselmann_outlier_info = {}
schiller_outlier_info = {}

#Green data-set IQR calculation
for g_atrib in green_attributes:
    g_values = green_data[g_atrib]

    g_set = []
    for value in g_values:
        g_set.append(float(value))

    n_samp = len(g_set)

    q75, q25 = np.percentile(g_set, [75,25])
    iqr = q75 - q25

    minThresh = q25 - 1.5*iqr
    maxThresh = q75 + 1.5*iqr

    green_outlier_info[g_atrib] = (minThresh, maxThresh, iqr)

print ('--- GREEN MIN/MAX/IQR ---')
print (green_outlier_info)

#Hinselmann data-set IQR calculation
for h_atrib in hinselmann_attributes:
    h_values = hinselmann_data[h_atrib]

    h_set = []
    for value in h_values:
        h_set.append(float(value))

    n_samp = len(h_set)

    q75, q25 = np.percentile(h_set, [75,25])
    iqr = q75 - q25

    minThresh = q25 - 1.5*iqr
    maxThresh = q75 + 1.5*iqr

    hinselmann_outlier_info[h_atrib] = (minThresh, maxThresh, iqr)

print ('--- HINSELMANN MIN/MAX/IQR ---')
print (hinselmann_outlier_info)

#Schiller data-set IQR calculation
for s_atrib in schiller_attributes:
    s_values = schiller_data[s_atrib]

    s_set = []
    for value in s_values:
        s_set.append(float(value))

    n_samp = len(s_set)

    q75, q25 = np.percentile(s_set, [75,25])
    iqr = q75 - q25

    minThresh = q25 - 1.5*iqr
    maxThresh = q75 + 1.5*iqr

    schiller_outlier_info[s_atrib] = (minThresh, maxThresh, iqr)

print ('--- SCHILLER MIN/MAX/IQR ---')
print (schiller_outlier_info)

#GLOBALS FOR COUNTING TOTAL OF OUTLIERS FOR EACH DATA-SET
green_lines_outliers = []
green_n_outliers = 0

hinselmann_lines_outliers = []
hinselmann_n_outliers = 0

schiller_lines_outliers = []
schiller_n_outliers = 0

#Counting outliers for green data-set
for g in green_attributes:
    lineCounter = 0
    values = green_data[g]

    for value in values:
        if (float(value) < green_outlier_info[g][0] or float(value) > green_outlier_info[g][1] and green_outlier_info[g][2] != 0):
            green_lines_outliers.append(lineCounter)
            green_n_outliers += 1

        lineCounter += 1


#Counting outliers for hinselmann data-set
for h in hinselmann_attributes:
    lineCounter = 0
    values = hinselmann_data[h]

    for value in values:
        if (float(value) < hinselmann_outlier_info[h][0] or float(value) > hinselmann_outlier_info[h][1] and hinselmann_outlier_info[h][2] != 0):
            hinselmann_lines_outliers.append(lineCounter)
            hinselmann_n_outliers += 1

        lineCounter += 1

#Counting outliers for schiller data-set
for s in schiller_attributes:
    lineCounter = 0
    values = schiller_data[s]

    for value in values:
        if(float(value) < schiller_outlier_info[s][0] or float(value) > schiller_outlier_info[s][1] and schiller_outlier_info[s][2] != 0):
            schiller_lines_outliers.append(lineCounter)
            schiller_n_outliers += 1

        lineCounter += 1

print ('\n')
print ('\n')

print ('--- GREEN COUNT OF OUTLIERS ---')
print (green_n_outliers)
print('--- GREEN COUNT OF LINES WITH OUTLIERS ---')
green_lines_outliers = np.unique(green_lines_outliers)
g = sorted(green_lines_outliers, reverse=True)
#print (g)
print (len(g))


print ('--- HINSELMANN COUNT OF OUTLIERS ---')
print (hinselmann_n_outliers)
print('--- HINSELMANN COUNT OF LINES WITH OUTLIERS ---')
hinselmann_lines_outliers = np.unique(hinselmann_lines_outliers)
h = sorted(hinselmann_lines_outliers, reverse=True)
#print (h)
print (len(h))


print ('--- SCHILLER COUNT OF OUTLIERS ---')
print (schiller_n_outliers)
print('--- SCHILLER COUNT OF LINES WITH OUTLIERS ---')
schiller_lines_outliers = np.unique(schiller_lines_outliers)
s = sorted(schiller_lines_outliers, reverse=True)
#print (s)
print (len(s))