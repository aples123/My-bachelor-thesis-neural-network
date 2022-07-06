# ANN-based method for the identification of proteins in proteomics experiments

-----------
Table of contents:
- [ About the data ](#data)
- [ Purpose of the project ](#project)
- [ Project structure ](#struct)
- [ Requirements ](#req)

-----------
<a name="data"></a>
## About the data
 
The data used in the project contains information about proteins, their measured mass spectra, sequences 
assigned to the spectra by the MASCOT database system, and a value of MASCOT build-in measure of matching the 
theoretical and experimental spectra. <br />
The dataset consists of two main parts. The first one, called *target*, are the real biological sequences 
matched by a database system to the measured samples. It can be a correct or incorrect match. The second one, called *decoy*, are fake, prepared sequences that the database system matched to the measured samples. <br />
-----------
<a name="project"></a>
## Purpose of the project
Unfortunately, like many other algorithms, the MASCOT measure does not grant a sufficient 
separation of correctly and incorrectly identified peptides, resulting in a large proportion 
of false identifications. <br />
In this project, I used a simple Multilayer Perceptron to create a new, better match score.
<br />
For that purpose, I had to extract a number of the *target* samples that with high probabilities are the correct matches and some *decoys*, which are the incorrect matches. To obtain that goal, I calculated a q-value for every sample and took only those *target* samples with a q-value not grater 
then a certain threshold (*qval_threshold* in the config file).

A new spectra match quality has granted much better separation of correctly and incorrectly identified peptides.
The significant difference is almost no overlap between low-rated *target* and *decoy* samples and highly rated *target* samples,
which was a huge problem in database measures. 

The project enables adjusting config parameters related to the training data selection and the training process.

-----------
<a name="struct"></a>
## Project structure 
The project contains following folders:
- scripts - main script + additional modules 
- resources - contains configurations, datasets, and optional saved models
- plots - created while saving plots  

-----------
<a name="req"></a>
## Requirements 
The project was created using the following versions of libraries:
- matplotlib 3.4.3
- numpy 1.20.3
- pandas 1.3.4
- tensorflow 2.7.0
- tqdm 4.62.3

It can be run via IDE or by running the script *main.py* using the terminal with an optional *filepath* argument, 
containing a path to a JSON file with the variables' configuration. The program uses the default configuration from the *config7b.json* file if the argument is not given.
