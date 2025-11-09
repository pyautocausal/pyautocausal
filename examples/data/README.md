The following datasets are used as examples

**1. California proposition 99 (`california_prop99.csv`)**
This data is from Abadie, Diamond, and Hainmueller (2010). The raw data is in MATLAB format from https://web.stanford.edu/~jhain/synthpage.html and is preprocessed here to a long panel, and saved as a `;` delimited CSV.

**2. Minimum wage increases (`mpdta.csv`)**
Data from Callaway and Sant’Anna (2021). Daata comes from states increasing their minimum wages on county-level teen employment rates. 

**3. Criteo Uplift modelling dataset**

A Large Scale Benchmark for Uplift Modeling Eustache Diemert, Artem Betlei, Christophe Renaudin; (Criteo AI Lab), Massih-Reza Amini (LIG, Grenoble INP). https://huggingface.co/datasets/criteo/criteo-uplift

Download using 
df = pd.read_csv("hf://datasets/criteo/criteo-uplift/criteo-research-uplift-v2.1.csv.gz")


