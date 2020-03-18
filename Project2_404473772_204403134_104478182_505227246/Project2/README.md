# Project 2: Clustering - Instructions to Run the Files
## Running the Scripts for Questions 1-11
Note that for this section, you need to have Jupyter installed.
The script is divided into multiple subsections that solve around 1 question each posed in the spec.
The testbench is written intermittently between sections to test and verify base functions.<br /><br />
IMPORTANT: Some sections require variables defined before them. Do run all sections in order.

## Running the Scripts for Question 12
The script for Question 12 is a separate Q12_regression.py script in the folder.
Provided all libraries are installed in you Python directory, this file should run on Python 2 or 3.<br /><br />
Simply enter the following command<br />
$ python Q12_regression.py<br /><br />
The regression will take a significant amount of time to run. To reduce time taken, decrease the number of epochs, or keep 'r' for testing purposes low.<br /><br />
The script outputs the best result of the regression up till a certain point in runtime. The output has 10 parameters in total: The first 5 elements in the list denote the metrics of the clustering in the order of [homogeneity_score, v_measure_score, completeness_score, adjusted_rand_score, adjusted_mutual_info_score].<br />
<br />
The last 3 elements reflect the [scale,log,order] of the scaling parameters. For more information on this, refer to Question 7 in the Jupyter notebook to see the implementation.<br />
<br />
Lastly the details about number of features and type of dimensionality reduction is given above each optimal parameter list.
For example, an output like this:<br />
> The new best parameters with 300, NMF<br />
> [0.47789955771326337, 0.5220625754592567, 0.575218942654573, 0.24848422255518163, 0.47619829597082236, 1, 1, 1]<br />

would indicate that number of features is 300, with NMF dimensionality reduction, [1,1,1] referring to log-scale scaling, and the metrics of the performance of clustering.
