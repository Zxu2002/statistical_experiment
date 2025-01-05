Statistical Experiment: Multi-Dimensional Probability Distribution
=========================================================

***Michaelmas Term 2024***

# Environment: Using pip
Note that the python version used for the project is 3.9.6. 
To install required packages:
```
pip install --no-cache-dir -r requirements.txt
```

---

# Project Structure 
The structure of the project is the following: 
```
.
├── README.md
├── code
│   ├── analyze_efg.py
│   ├── bootstrap_errors.npy
│   ├── bootstrap_lambda_values.npy
│   ├── bootstrape.py
│   ├── generate_sample.py
│   ├── sweight_errors.npy
│   ├── sweight_lambda_values.npy
│   ├── sweight_projection.py
│   └── total_model.py
├── report
│   ├── e_bootstrape_plot.png
│   ├── f_sweight_plot.png
│   ├── g_comparison_plot.png
│   ├── q3_1d_projection.png
│   └── q3_2d_plot.png
└── requirements.txt

```
In the code folder, the ```total_model.py``` file contains definition for the multi-dimensional probability distribution. The ```generate_sample.py``` generates sample from the distirbution and performs an extended maximum likelihood fit for the parameters. The files ```bootstrape.py``` and ```sweight_projection.py``` perfoms multi-dimensional likelihood fit and weighted fit repspectively, generating results in the npy files. The python file ```analyze_efg.py``` analyzes the data in npy files and produce plots to answer question (e), (f) and (g). 
All the plots are saved in the report folder. 
To execute each file, please run the files directly. 

# Acknowledgement
The code was developed with the assistance from Anthropic’s Claude 3.5 Sonnet. Specifically, Claude was used to assist with debugging Python code implementations of statistical analysis methods. All AI-generated suggestions were manually reviewed.
