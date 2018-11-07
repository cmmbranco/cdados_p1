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

######################################
# PERCENTAGE OF OUTLIERS PER FEATURE #
######################################
green_outliers_per_feature = {}
hinselmann_outliers_per_feature = {}
schiller_outliers_per_feature = {}

# Counting outliers per feature for green data-set
for g in green_attributes:
    counter = 0
    values = green_data[g]

    for value in values:
        if (float(value) < green_outlier_info[g][0] or float(value) > green_outlier_info[g][1] and green_outlier_info[g][2] != 0):
            counter +=1
    # Inserir o numero de outliers na feature
    green_outliers_per_feature[g] = counter

# Convert into percentage and print
for feature in green_outliers_per_feature:
    green_outliers_per_feature[feature] = (green_outliers_per_feature[feature]/98) * 100.0

print ('-- GREEN % OF OUTLIERS PER FEATURE --')
print (green_outliers_per_feature)

# Counting outliers per feature for hinselmann data-set
for h in hinselmann_attributes:
    counter = 0
    values = hinselmann_data[h]

    for value in values:
        if (float(value) < hinselmann_outlier_info[h][0] or float(value) > hinselmann_outlier_info[h][1] and hinselmann_outlier_info[h][2] != 0):
            counter += 1
    # Inserir o numero de outliers na feature
    hinselmann_outliers_per_feature[h] = counter

# Convert into percentage and print
for feature in hinselmann_outliers_per_feature:
    hinselmann_outliers_per_feature[feature] = (hinselmann_outliers_per_feature[feature]/97) * 100.0

print ('-- HINSELMANN % OF OUTLIERS PER FEATURE --')
print (hinselmann_outliers_per_feature)

# Counting outliers per feature for schiller data-set
for s in schiller_attributes:
    counter = 0
    values = schiller_data[s]

    for value in values:
        if (float(value) < schiller_outlier_info[s][0] or float(value) > schiller_outlier_info[s][1] and schiller_outlier_info[s][2] != 0):
            counter += 1
    # Inserir o numero de outliers na feature
    schiller_outliers_per_feature[s] = counter

# Convert into percentage and print
for feature in schiller_outliers_per_feature:
    schiller_outliers_per_feature[feature] = (schiller_outliers_per_feature[feature]/92) * 100.0

print ('-- SCHILLER % OF OUTLIERS PER FEATURE --')
print (schiller_outliers_per_feature)
