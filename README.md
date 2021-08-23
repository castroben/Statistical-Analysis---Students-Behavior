DATASET DESCRIPTION 
The data set analyzed in the project is composed of 29,305 entries describing the video-watching 
behaviors of 3876 students over a total of 93 videos. The entries also contain the results the students 
obtained on their first attempt on the quizzes corresponding to the videos. Each video has a unique ID, 
which is a number between 0 and 92. Each student is characterized by a unique user ID, and each entry 
is unique for the behaviors/score of a particular student on a particular video/quiz. There was no 
specification on the number of videos a student had to watch. Therefore, if a student only watched one 
video and completed its corresponding quiz, the user ID corresponding to the student and the 
corresponding behaviors/score appear only once in the data set. Similarly, if a student watched multiple 
videos and completed the corresponding quizzes, the user ID corresponding to the student and the 
corresponding behaviors/score appear multiple times in the data set. 
There are 9 video-watching metrics in the data set. The metrics are defined as follows:

1. fracSpent - the fraction of time the student spent watching the video relative to the length of 
the video.
2. fracComp - the fraction of the video the student watched. This number ranges between 0 and 1 
where 0 corresponds to watching none of the video and 1 corresponds to watching the video 
completely.
3. fracPlayed – no definition provided.*
4. fracPaused - the fraction of time the student spent paused on the video, relative to the length of 
the video.
5. numPauses - the number of times the student paused the video.
6. avgPBR - the average playback rate that the student used while watching the video, ranging 
between 0.5x and 2.0x.
7. stdBPR – no definition provided.*
8. numRWs - the number of times the student skipped backwards (rewind) in the video.
9. numFFs - the number of times the student skipped forward (fast forward) in the video.
* Even though one might think the labels of these metrics are self-descriptive, since no concrete 
descriptions were provided for these metrics, we did not include them in our analysis to avoid any 
corruption in the data.

ANALYSIS METHODS 
For this project we considered three main analyses questions:
1. How well can the students be grouped based on their video-watching behaviors?
2. Can the video-watching behaviors of a student be used to predict the student’s performance on 
video quizzes (average score across n videos)?
3. Can the behaviors of a student on a particular video be used to predict the student’s score on 
the corresponding video quiz question?
1. 
For this first analysis question we decided to implement a clustering algorithm on a subset of the data 
set. Specifically, we implemented the k-means algorithm. This is an iterative centroid-based algorithm 
which finds a simple structure in the data. For this analysis we only considered the students that 
completed at least five of the videos. We determined that a student completed at least five videos if the 
user ID corresponding to the student appeared in at least five entries and those entries had a fracComp
score above 0.98. For this analysis we considered all seven metrics defined above. The metrics were 
averaged for each student to perform the analysis. Finally, we implemented the algorithm for a range of 
clusters between 2 and 15 to find the optimal number of clusters into which to divide the data.
We chose to implement the k-means algorithm for this analysis because it allowed us to find a simple 
structure given a multi-dimensional data set. We also chose this technique because it allowed us to call 
the algorithm with a varying number of centroids to find the optimal number of centroids. Additionally, 
this model does not require an assumption about any preexisting model in the data. We evaluated the 
clustering models by computing the distortions in each of the models for the range of centroids we 
defined. Additionally, we computed the Calinski-Harabasz score for each model given a number of 
centroids to further identify the optimal number of centroids. We expected this analysis to tell us if the 
students can be naturally grouped into some number of clusters, and what that number would be.
2. 
For this analysis question we decided to implement Ridge regression with cross validation on a subset of 
the data set. We considered only the students that completed at least half of the quizzes in our analysis. 
We determined that a student completed at least half of the quizzes if the user ID corresponding to the 
student appeared more than 46 times in the data set. All the seven metrics defined above were 
considered in our analysis. The seven metrics were averaged for each student to create a feature matrix 
of all students. The score entries for each student were also averaged to create a target vector of all 
students. For the Ridge regression analysis, we chose a range of regularization parameters in a 
logarithmic space between 1 and 2. We considered a total of 100 regularization parameters in this 
range. After performing the Ridge regression analysis with cross validation, we obtained the most 3
optimal model for the data. We evaluated this model by computing the coefficient of determination for 
the model obtained. 
We decided to use Ridge regression with cross validation because this method provides a way to obtain 
an optimal linear model with a regularization parameter. With the regularization parameter, we can 
compute a model that best fits the data but avoids overfitting. Ridge regression also provides a set of 
coefficients that allowed us to infer the effects of each behavior on the average performance of each 
student. From the analysis we expected to identify whether there exists a reliable model for predicting 
the performance of students of the video quizzes given their behaviors. 
3.
For this analysis question, we decided to implement a logistic regression analysis with cross validation 
on a subset the data set. We chose a logistic regression analysis since the score a student received on a 
particular video quiz could only take two values, zero or one. Due to the nature of the scores, we 
identified that the problem of predicting the scores of students could be shaped into a classification 
problem. Logistic regression was the ideal tool for this analysis because we could classify each entry in 
one of two categories. For this analysis we considered the same metrics as in the previous two analyses. 
However, in this analysis we did not perform any averaging, rather we analyzed the behaviors in each 
entry and the respective score on the quiz. From the analysis we expected to identify whether there 
exists a reliable model for predicting the score of a student on a particular video quiz given his behaviors 
on that video.
