SMO
===

A variation of Sequential Minimal Optimization algorithm implementation

########## SVMs for binary classification ###########
The file smo_impr.cc can be used for model learning and inferencing.
cv_smo_impr.cc differs from smo_impr.cc chiefly in its main function. This
source code can be used to get the k-fold cv error.

To produce the executable from smo_impr.cc:
g++ -o smo_impr smo_impr.cc -lm

To get the executable from cv_smo_impr.cc:
g++ -o cv_smo_impr cv_smo_impr.cc -lm

To know the usage of the executables, just type the executable name in
the command line. For example, typing smo_impr (produced above) results
in:

Usage:
num_training_samples, tr_file dim, C, sigmasqr/degree test_samples, Test_file kernelchoice(1:RBF,2:dp 3:poly) file format:sparse(1)/dense(0)

smo_impr takes 9 arguments. They are:

1) #training sample - number of training samples
2) tr_file - name of the training file
3) dim - dimension of the dataset
4) C - hyperparameter
5) sigmasqr/degree - hyperparameter, for the kernel function
6) #test samples - number of test samples (when you don't have one, just
        pass the training file itself to get the training set error)
7) test_file - name of the test dataset
8) kernel choice - 1 indicates Radial Basis function, 2 - linear kernel, 
        3- polynomial kernel
9) file format of the datasets (see below)

All arguments are compulsory. For linear kernel, just pass any value to
sigmasqr argument.

Example input command:
smo_impr 194 datasets/twospirals.txt 2 0.04 0.5 194 datasets/twospirals.txt 1 0

A sample output (for a dataset with dim > 2):

Time in seconds:1.27276
Thresh:0.3793
The nBsvs:110 Bsvs:237 Zsvs:336
Number of kernel evaluations * 1e-6: 9.877583
The error rate:3.51

For a dataset with 2 dimensions, the code outputs additional information
that can be used for plotting support vectors and errors.

############## CROSS-VALIDATION experiment ###########
Typing cv_smo_impr in command line results in:

Usage:
num_training_samples, tr_file dim, sigmasqr/degree kernelchoice(1:RBF,2:dp 3:poly) file format:sparse(1)/dense(0) C-array file K (cv fold size)

cv_smo_impr takes 8 arguments. Only the arguments that are different from
arguments of smo_impr are described below:

1) C-array file - input file that contains list of C values. Should have
only a single column of floating point number, without unwanted spaces.
2) K (cv fold size) - number of folds/divisions required out of the
dataset. Common value is 10 for large datasets. For small datasets, pass
large values upto the dataset size.

All arguments are compulsory. For linear kernel, just pass any value to
sigmasqr argument.

Example input command:
cv_smo_impr 2477 datasets/w1a 300 10.0 1 1 webcollection/c_web 25

A sample output:

C Toterror
0.020 3.670257
0.040 3.670257
0.060 3.523198
0.100 3.229080
0.200 3.229080
0.400 3.229080
0.500 3.376139
0.700 3.376139
1.000 3.376139
2.000 3.382353
3.000 3.382353


############ DATASET FORMAT#############
One line of a dense dataset looks as follows:
-1 -6.500000 -0.000000

-1 is the class specification of the point, followed by actual values from
each dimension. Only +1 and -1 are allowed class specifications.

One line of a sparse dataset looks as follows:
+1 3:1 11:1 14:1 19:1 39:1 42:1 55:1 64:1 67:1 73:1 75:1 76:1 80:1 83:1 

+1 is the class specification of the point. Each entry following it is of
the form columnidx:val. For more details, please refer, LIBSVM dataset
collection.

Caution:
The datasets cannot have additional spaces or escape characters other than
required single space separation between entries in each row and one newline
character at the end of each line.

Future work:
1) Saving the model file separately and do testing on choice basis
2) Accept sigmasqr array for CV-routine
3) Accept variable number of arguments for each command line
4) Divide the source file into multiple files - one file for SVM related
functions, one file for input and output etc.
5) Create Makefile for automation
6) Do flexible file reading, not tied to the number of additional
spaces or escape characters
7) calculate the number of training samples and dim instead of asking from
user
8) Combine cross-validation routine into main program and do CV in
choice-based manner.

############## MATLAB #############
For all matlab codes, the header of the file gives information about
parameters to the script.
linpts.m is to produce Figure 1 in the report.
quadplot.m is to produce Figure 2 in the report.
checker.m corresponds to Figure 6 in report.
