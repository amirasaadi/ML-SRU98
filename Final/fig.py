# import numpy as np
# from matplotlib import pyplot as plt
# # res knn bayes DT SVM LR LDA MLP
# labels = ['SVM','MLP','KNN','LogisticRegression' ,'LDA','decisionTree' , 'Bayes']
# x = np.arange(len(labels))  # the label locations
# width = 3
# res = [0.962,0.944,0.936,0.933,0.922,0.855,0.826]
# fig, ax = plt.subplots()
# ax.bar(labels,res,color='green')
# ax.set_xticklabels(labels)
# ax.set_ylabel('Performance')
# fig.autofmt_xdate()
# plt.show()

# import matplotlib
# import matplotlib.pyplot as plt
# import numpy as np
#
# nn3= np.array([98.0, 93.0, 95.5, 92.0, 94.0, 97.0, 95.0, 96.5, 97.0, 96.0])
# nn3= nn3.mean()
# bayes=np.array([52.5, 78.5, 58.5, 84.0, 60.5, 74.5, 71.5, 65.0, 63.0, 60.5])
# bayes=bayes.mean()
# labels = ['KNN', 'Bayes', 'MLP']
# men_means = [nn3, bayes, 86.7]
# women_means = [93.6, 82.6, 94.4]
#
# x = np.arange(len(labels))  # the label locations
# width = 0.35  # the width of the bars
#
# fig, ax = plt.subplots()
# rects1 = ax.bar(x - width/2, men_means, width, label='written from scratch')
# rects2 = ax.bar(x + width/2, women_means, width, label='scikit-learn')
#
# # Add some text for labels, title and custom x-axis tick labels, etc.
# ax.set_ylabel('Scores')
# ax.set_xticks(x)
# ax.set_xticklabels(labels)
# ax.legend()
#
#
# def autolabel(rects):
#     """Attach a text label above each bar in *rects*, displaying its height."""
#     for rect in rects:
#         height = rect.get_height()
#         ax.annotate('{}'.format(height),
#                     xy=(rect.get_x() + rect.get_width() / 2, height),
#                     xytext=(0, 3),  # 3 points vertical offset
#                     textcoords="offset points",
#                     ha='center', va='bottom')
#
#
# autolabel(rects1)
# autolabel(rects2)
#
# fig.tight_layout()
#
# plt.show()

import numpy as np
p = np.asarray([[0.6, 0.4],[0.5 ,0.5]])
p2 = p@p
p4 = p2@p2
print(p4)