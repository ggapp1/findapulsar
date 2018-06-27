import numpy as np 
import pandas as pd 

from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
data = pd.read_csv('dataset/pulsar_stars.csv')
#drop missing values
data.dropna()
print data.columns

data = data.rename(columns={' Mean of the integrated profile':"mean_profile",
       ' Standard deviation of the integrated profile':"std_profile",
       ' Excess kurtosis of the integrated profile':"kurtosis_profile",
       ' Skewness of the integrated profile':"skewness_profile", 
        ' Mean of the DM-SNR curve':"mean_dmsnr_curve",
       ' Standard deviation of the DM-SNR curve':"std_dmsnr_curve",
       ' Excess kurtosis of the DM-SNR curve':"kurtosis_dmsnr_curve",
       ' Skewness of the DM-SNR curve':"skewness_dmsnr_curve",
       })
plt.figure(figsize=(12,6))
plt.subplot(121)
sns.set(style="darkgrid")
ax = sns.countplot(x = data["target_class"])
plt.title("Pulsar/not pulsar")
#plt.show()


correlation = data.corr()
plt.figure(figsize=(9,7))
sns.heatmap(correlation,annot=True,cmap=sns.light_palette("green"),linewidth=2,edgecolor="k")
plt.title("CORRELATION BETWEEN VARIABLES")
#plt.show()


compare = data.groupby("target_class")[['mean_profile', 'std_profile', 'kurtosis_profile', 'skewness_profile',
                                        'mean_dmsnr_curve', 'std_dmsnr_curve', 'kurtosis_dmsnr_curve',
                                        'skewness_dmsnr_curve']].mean().reset_index()


compare = compare.drop("target_class",axis =1)
sns.set(style="darkgrid")
compare.plot(kind="barh",width=.6,figsize=(13,6))
plt.title("media dos valores")

compare1 = data.groupby("target_class")[['mean_profile', 'std_profile', 'kurtosis_profile', 'skewness_profile',
                                        'mean_dmsnr_curve', 'std_dmsnr_curve', 'kurtosis_dmsnr_curve',
                                        'skewness_dmsnr_curve']].std().reset_index()
compare1 = compare1.drop("target_class",axis=1)
compare1.plot(kind="barh",width=.6,figsize=(13,6))
plt.title("desvio padrao dos valores")
plt.show()