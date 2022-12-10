# CMPT353 Final Project

In this project we will analyze patterns within steps taken to see if we can identify any injuries through our data.

# Required Libraries

All libraries should be included in Anaconda.

pandas, numpy, matplotlib, statsmodels, scipy, sklearn

# Commands and Arguments

```bash
$python project.py data/*.csv output
```
project.py takes 2 arguments, a data directory with csv data giles and an output directory. Before running, ensure project has a data directory with csv files inside.


# Files Produced

After running the code, this program will create the following output for each data csv file: a walk datagraph and a frequency graph in the output directory, as well as print Mann-U p-values on the terminal.
