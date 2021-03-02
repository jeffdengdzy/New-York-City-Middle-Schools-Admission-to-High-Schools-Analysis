# -*- coding: utf-8 -*-
"""
Created on Mon Dec 21 15:14:09 2020
DS-112 Final Project

@author: Ziyi Deng
@netid: zd674
@N-number: N19360164
"""

# For the dimension reduction, data cleaning and data transformation, I apply
# dimension reduction to the L-Q columns and V-X columns in question 4. I extract
# a main factor component for each This will also be used in question 5 and 8. 
# As for data cleaning and transformation, I create a new copy of the original
# dataframe for each coding question except question 1,2,3 and 7. In this way, I'll
# customize the action of dropping or imputing dataframe for each question and
# to reserve the amount of data for each question as much as possible instead
# of abandoning too much detail of the data by an overall dropping. In brief, 
# the dropping or the imputing of data is different for each question.  


import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import statsmodels.api as sm
from sklearn import linear_model
from scipy import stats

df = pd.read_csv("middleSchoolData.csv")


"Question 1"

# Compute the correlation between admission number and application number. 
corr_app = np.corrcoef(df["acceptances"], df["applications"])
corr_app[0,1]
# The correlation between them is 0.8017, which demonstrate a strong positive
# linear relationship. 

# Do the scatter plot
plot1 = plt.figure(1)
x = df["applications"]
y = df["acceptances"]
plt.scatter(x,y)
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='red')
plt.xlabel("Number of applications")
plt.ylabel('Number of acceptances')
plt.title("Correlation between the number of applications and admissions to HSPHS")
plot1.show()
# We notice that the data is more concentrated in the bottom left part while 
# more sparsely distributed in the upper right part. The positive linear 
# relationship between x and y can be easily observed. 



"Question 2"

df["application_rate"] = df["applications"] / df["school_size"]
df["application_rate"] = df["application_rate"].fillna(0)
corr_app_rate = np.corrcoef(df["acceptances"], df["application_rate"])
cod_app = corr_app[0,1] ** corr_app[0,1]
cod_app_rate = corr_app_rate[0,1] ** corr_app_rate[0,1]
cod_app
cod_app_rate
# The coefficient of determination between application and acceptance is 0.8376. 
# The coefficient of determination between application rate and acceptance
# is 0.7583. 
# Since the coefficient of determination of the application number regressed to
# admission number is bigger than that of the application rate, we determine
# that the raw number of applications is a better predictor of admission to 
# HSPHS than application rate. 



"Question 3"

df["acceptance_rate"] = df["acceptances"] / df["school_size"]
best = df.loc[df["acceptance_rate"] == max(df["acceptance_rate"])]
best["school_name"]
# The Christa Mcauliffe School has the best *per student* odds of 
# sending someone to HSPHS. 



"Question 4"

LQ = ["rigorous_instruction", "collaborative_teachers", "supportive_environment"
      , "effective_school_leadership", "strong_family_community_ties", 
      "trust"]
VX = ["student_achievement", "reading_scores_exceed", "math_scores_exceed"]
LQ_VX = LQ + VX

# Create dataframe 4 that drops all NA in L-Q and V-X columns. 
df4 = df.dropna(subset = LQ_VX)
df4_LQ = df4[LQ]

# Run the PCA for L-Q columns
pca = PCA()
zscored_LQ = stats.zscore(df4_LQ)
pca.fit(zscored_LQ)
eig_vals_LQ = pca.explained_variance_
loadings_LQ = pca.components_
rotated_data_LQ = pca.fit_transform(zscored_LQ)
covar_explained_LQ = eig_vals_LQ/sum(eig_vals_LQ)*100

# Run the PCA for V-X columns
df4_VX = df4[VX]
zscored_VX = stats.zscore(df4_VX)
pca.fit(zscored_VX)
eig_vals_VX = pca.explained_variance_
loadings_VX = pca.components_
rotated_data_VX = pca.fit_transform(zscored_VX)
covar_explained_VX = eig_vals_VX/sum(eig_vals_VX)*100

# Make the screeplot
num_classes_LQ = 6
num_classes_VX = 3
plot4 = plt.figure(4)
plt.plot(np.linspace(1,num_classes_LQ,num_classes_LQ),eig_vals_LQ, color="blue")
plt.plot(np.linspace(1,num_classes_VX,num_classes_VX),eig_vals_VX, color="red")
plt.xlabel('Principal component')
plt.ylabel('Eigenvalue')
plt.plot([0,num_classes_LQ],[1,1],color='red',linewidth=1)
plot4.show()
# From the screeplot, we can find that only one factor in L-Q variables and 
# one factor in V-X variables have eigensum over 1. Therefore, next we'll only
# extract one component from each group to do the correlation. 

# Extract the component from L-Q group
pca = PCA(n_components=1)
pca.fit(zscored_LQ)
principalLQ = pca.fit_transform(zscored_LQ)
principalLQdf = pd.DataFrame(data = principalLQ
             , columns = ['principal component L-Q'])

# Extract the component from V-X group
pca.fit(zscored_VX)
principalVX = pca.fit_transform(zscored_VX)
principalVXdf = pd.DataFrame(data = principalVX
             , columns = ['principal component V-X'])

# Correlate the two components
pca_corr = np.corrcoef(principalLQdf['principal component L-Q'], 
                   principalVXdf['principal component V-X'])
pca_corr[0,1]
# The correlation between n how students perceive their school and how the
# school performs on objective measures of achievement is -0.3674 using PCA
# dimension reduction. This value demonstrates a moderate negative relationship
# as -0.7 < r < -0.3. 



"Question 5"

# We'll perform a null test between the school size and the objective 
# measurement of achievement. The null hypothesis states that the school
# size doesn't influence the objective achievement. 

# Create dataframe 5 that drops NA in school size column and V-X columns that
# represent the objective measurement of achievement. 
df5 = df.dropna(subset = ["school_size","student_achievement",
                          "reading_scores_exceed", "math_scores_exceed"])

# Generate the PCA for objective achievement based on the particular subgroup
# of droppings of dataframe 5. 
df5_VX = df5[VX]
zscored_VX = stats.zscore(df5_VX)
pca = PCA(n_components=1)
pca.fit(zscored_VX)
principalVX = pca.fit_transform(zscored_VX)
df5["objective_achievement"] = principalVX

# Transform the school size in to categorical variable. The schools with school
# size above the median are assigned to 1 while those belwo the median are
# assigned to 0. 
school_size_median = np.median(df5["school_size"])
df5['school_size_binary'] = np.where(df5['school_size'] >
                                     school_size_median, 1, 0)
# Perform the t test
model = sm.OLS.from_formula("objective_achievement ~ school_size_binary", data=df5)
result = model.fit()
result.summary()
# Since the p value 0 is smaller than 0.05, at the 95% confidence level, we
# reject the null hypothesis. We conclude that the school size will influence
# the objective achievement of students. 



"Question 6"

# Select the per student spending and admission to HSPHS to do this question

# Create the dataframe 6 that imputes all missing "per_pupil_spending" values
# with its mean. We didn't choose to drop the missing values so that we won't 
# drop all data of charter school and cause bias. 
df6 = df
df6["per_pupil_spending"] = df6["per_pupil_spending"].fillna(np.mean(df6["per_pupil_spending"]))

# Correlate admission and per student spending
corr_spending = np.corrcoef(df6["acceptances"], df6["per_pupil_spending"])
corr_spending[0,1]
# The correlation between admission and per student spending is -0.3315, which
# demonstrate a moderate negative relationship as -0.7 < r < -0.3. Therefore, 
# we conjecture that per student spending will impact the admission negatively. 

# Do the scatter plot
plot6 = plt.figure(6)
x = df6["per_pupil_spending"]
y = df6["acceptances"]
plt.scatter(x,y,color = "purple")
plt.plot(np.unique(x), np.poly1d(np.polyfit(x, y, 1))(np.unique(x)), color='green')
plt.xlabel(" Per student spending, in $")
plt.ylabel('Number of acceptances')
plt.title("Correlation between the per student spending and admissions to HSPHS")
plot6.show()
# From the plot we observe that most of the HSPHS acceptance numbers are between
# 0 and 50. Several high acceptance numbers are scattered when per student spending
# is lowe. However, when per student spending increases, almost all of the 
# acceptance numbers are around 0. 

# Perform a t test between admission and per student spending
spending_median = np.median(df6["per_pupil_spending"])
df6['per_pupil_spending_binary'] = np.where(df6['per_pupil_spending'] >
                                     spending_median, 1, 0)
model = sm.OLS.from_formula("acceptances ~ per_pupil_spending_binary", data=df6)
result = model.fit()
result.summary()
# From the result table, we reject the null hypothesis as the p value is 0. 
# Therefore, we further confirm that per student spending will impact the 
# admission to HSPHS. 



"Question 7"

# Sort the acceptance number in descending order
acceptance_sort = -np.sort(-df['acceptances'])

# Compute the probability density distribution and cumulative density distribution
pdf = (acceptance_sort / sum(acceptance_sort)) * 100
cdf = np.cumsum(pdf)

# Locate the threshold index and compute the proportion
cdfFindMin = abs(cdf - 90)
index_90_percentile = cdfFindMin.argmin()
proportion = index_90_percentile / len(cdf)
proportion
# 20.37% of schools accounts for 90% of all students accepted to HSPHS. 

# Plot the bar graph of probability density distribution. 
# The orange columns represent the top 20.37% schools that accounts for 90% admissions
# while the blue ones repressent the remaining 79.63% that contribute 10%. 
x_axis = np.linspace(1,len(pdf),num=len(pdf))
plot7 = plot1 = plt.figure(7)
bar = plt.bar(x_axis,pdf)
for i in range(index_90_percentile): 
    bar[i].set_color("orange")
for j in range(index_90_percentile,len(pdf)): 
    bar[j].set_color("blue")
plt.xlabel("Order of schools from higher admission to HSPHS to lower")
plt.ylabel("Proportion of admitted students")
plt.title("Probability density distribution of schools account for HSPHS admission")
plot7.show()



"Question 8"

# Create the dataframe 8 that imputes all missing "per_pupil_spending" values
# and "avg_class_size" values with their means respectively to avoid dropping 
# all of the charter school data. After charter school data are "protected”， 
# we drop all of rows with missing value. 

df8 = df
df8["per_pupil_spending"] = df8["per_pupil_spending"].fillna(np.mean(df8["per_pupil_spending"]))
df8["avg_class_size"] = df8["avg_class_size"].fillna(np.mean(df8["avg_class_size"]))
df8 = df8.dropna()

# Run the PCA for "average_student_climate" and "objective achievement" for 
# datafrae 8
df8LQ = df8.loc[:, LQ].values
df8LQ = StandardScaler().fit_transform(df8LQ)
pca = PCA(n_components=1)
principalComponentsLQ = pca.fit_transform(df8LQ)
df8["average_student_climate"] = principalComponentsLQ

df8VX = df8.loc[:, VX].values
df8VX = StandardScaler().fit_transform(df8VX)
pca = PCA(n_components=1)
principalComponentsVX = pca.fit_transform(df8VX)
df8["objective_achievement"] = principalComponentsVX

# The columns of percent of races look clumsy. However, because they are mutual
# exlusive, it's not appropriate for us to do the pca for them. Therefore, I
# decide to convert them to a set of racial diversity that accounts for the
# race proportion distribution in each school. 

# Compute the standard deviation of the six data of race percent for each schoool. 
mean = (df8["asian_percent"]+df8["black_percent"]+df8["hispanic_percent"]
                       +df8["multiple_percent"]+df8["white_percent"]) / 5
df8["racial_diversity"] = (((df8["asian_percent"]-mean)**2 
                            + (df8["hispanic_percent"]-mean)**2
                            +(df8["black_percent"]-mean)**2 
                            +(df8["multiple_percent"]-mean)**2
                            +(df8["white_percent"]-mean)**2) / 5)**(1/2)

# The less deviated the number is, the more diverse the schools are. 
# So we make the result reciprocal to denote the level of diversity: 
df8["racial_diversity"] = 1 / df8["racial_diversity"]

# Since all factors are involved, we'll normalize all of the data in the event of
# some of the variables might be skewed. 
df8 = df8.iloc[:,2:]
df8_normalize = (df8-df8.mean())/df8.std()
df8_normalize = (df8-df8.min())/(df8.max()-df8.min())

# implement the multiple regression for HSPHS admission
regression1_factor = ["per_pupil_spending","avg_class_size","racial_diversity"
                      ,"average_student_climate",
                      "disability_percent","poverty_percent","ESL_percent",
                      "school_size","objective_achievement", "applications"]

regression1_x = df8_normalize[regression1_factor]
regression1_y = df8["acceptances"]

regr = linear_model.LinearRegression()
regr.fit(regression1_x, regression1_y)

regression1_x = sm.add_constant(regression1_x)
model = sm.OLS(regression1_y, regression1_x).fit()
predictions = model.predict(regression1_x)
print_model1 = model.summary()
print_model1
# From the table, we find that the application number is the most important characteristic 
# in terms of sending students to HSPHS as its coefficient is the biggest. Also, we
# notice that the school size matter too. 

# multiple regression for objective measures or achievement 
regression2_factor = ["per_pupil_spending","avg_class_size","racial_diversity"
                      ,"average_student_climate",
                      "disability_percent","poverty_percent","ESL_percent",
                      "school_size", "applications"]

regression2_x = df8_normalize[regression2_factor]
regression2_y = df8["objective_achievement"]

regr = linear_model.LinearRegression()
regr.fit(regression2_x, regression1_y)

regression1_x = sm.add_constant(regression2_x)
model = sm.OLS(regression1_y, regression2_x).fit()
predictions = model.predict(regression2_x)
print_model2 = model.summary()
print_model2
# From the table, we find that the application number is the most important 
# characteristic in terms of achieving high scores on objective measures of achievement. 
# Also, the school size matters. 



"Question 9"

# In summary, the admission to HSPHS and application number is storngly postively
# correlated; the raw number of applications is a better predictor of admission
# to HSPHS than application rate; The Christa Mcauliffe School has the best 
# *per student* odds of sending someone to HSPHS; there's a moderate negative relationship 
# between how students perceive their school and how the school performs on objective 
# measures of achievement; the school size will influence the objective 
# achievement of students; there's evidence that per student spending will impact the 
# admission to HSPHS; 20.37% of schools accounts for 90% of all students
# accepted to HSPHS; the application number is the most important characteristic
# in terms of both sending students to HSPHS and achieving high scores on objective 
# measures of achievement. 

# From the table 1 that we got from question 8, it is shown that the application 
# number to be most relevant in determining acceptance of their students to HSPHS. 
# Also, from some varaibles in the table with big-enough coefficient and smaller-
# than-0.05 p value, we can say that racial diversity, poverty percent, school size, 
# and objective achievement influence admission to HSPHS as well. While the influence
# of application number and objective achievement are postive, the impact of 
# racial diversity, poverty percent, school size are negative. 



"Question 10"

# I'd suggest Department of Education to reduce the average school size. To 
# accomplish this, Department of Education could found more schools, and they 
# will attract the new students and some students from the crowded schools. 
# In this way, the average school size of schools in the city will be relieved. 
# As school size functions negatively on either admission to HSPHS or objective
# measures of achievement, reducing school size will help the schools improve
# these two parameters. 

# In addition, even though the factor of application number seems so alluring
# as of it's coefficient number in both tables, as for the level of Department
# of Education, which faces every school in the city, I don't suggest 
# Department of Education to ask all schools to encourage more students to 
# apply to HSPHS. This is because the total admission number of HSPHS holds 
# constant and won't be affected by the total number of applications from the 
# city. The increasing amount of applications will only reduce the admission 
# rate, which will cause cut-throat competition. For this reason, I don't 
# recommend Department of Education to encourage the HSPHS application citywide. 





