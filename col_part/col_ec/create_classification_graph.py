import matplotlib.pyplot as plt
import numpy as np


######################
# Parameters to pass K #
######################

sensitivity = [82.38095238095238, 83.0952380952381, 77.61904761904761]
specificity = [69.04761904761905, 66.9047619047619, 73.57142857142857]
accu = [75.71428571428571, 74.99999999999997, 75.5952380952381]
nrforest = [5, 15, 30]

plt.plot(nrforest, sensitivity, 'g', label=r'Mean Sensitivity')
plt.plot(nrforest, specificity, 'r', label=r'Mean Specificity')
plt.plot(nrforest, accu, 'b', label=r'Mean Accuracy')

plt.xlabel('Max_Features')
plt.ylabel('Performance(%)')
plt.title('Performance Chart for CART (GREEN)')
plt.legend(loc="lower right")
plt.show()

#data to plot
#n_groups = 3

#create plot
#fig, ax = plt.subplots()
#index = np.arange(n_groups)
#bar_width = 0.25
#opacity = 0.5

#sensivity = [63.80, 86.94, 80.95]
#specificity = [85.00, 52.36, 66.19]
#acc = [73.066, 92.390, 91.937]

#plt.bar(index, sensivity, bar_width, color='g', label=r'Mean Sensitivity')
#plt.bar(index + bar_width, specificity, bar_width, color='r', label=r'Mean Specificity')
#plt.bar(index, acc, bar_width, color='orange', label=r'Mean Accuracy')


#plt.xlabel('Measures')
#plt.ylabel('Performance(%)')
#plt.title('Performance Chart for NB')
#plt.legend(loc="lower right")
#plt.xticks(index, ('Sensitivity', 'Specificity','Accuracy'))
#plt.show()