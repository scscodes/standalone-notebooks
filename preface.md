# Medical Model Monitor | M3
##### Capstone 4: Applied Model Application
---

## Use Case & Scenario
- **Problem Statement:**
  Hospitals struggle to continuously monitor patients' vital signs and promptly identify those at risk of adverse events. Complex data and environmental factors can delay intervention, increase risk, and incur higher expense.

- **Solution:**
  A mobile and desktop application that notifies staff of patients needing assistance. From a technical perspective, this involves training a range of models to establish a baseline of knowledge, creating a prediction for values based on historical analysis, and managing real-time inputs for validation.


---

## Approach
Considering the scope, data analysis will be logically separate from model development. This will help reduce overhead and visual complexity. As such, please make sure you're reviewing the correct notebook:

- Data Collection & Analysis `<- you are here`
- Model Development


## Notebook Structure
- Prerequisites; imports, generic functions
- Data import and extraction
- Analysis of raw data
- Cleaning and generalizations
- Analysis of transformed data
---
#%% md
## Data Authorization and Access
Research-grade medical data is considered public, but retains certain safeguards to deter improper use and mitigate risk. This introduced some delay as various steps were completed, and requests were processed by third party teams and systems. These prerequisites were anticipated though, as outlined in the project proposal.

To summarize, the following steps were completed in order to gain access to the `MIMIC IV` dataset:
1. Registration on PhysioNet website
2. PhysioNet application review and approval; use case and reference evaluation
3. Training Completion; CITI 'Data or Specimens Only Research Training'
4. Code of Conduct agreement
5. Credentialed Health Data Use Agreement (per dataset)

After all steps were completed, access was granted to `credentialed datasets`, to include the `MIMIC IV` dataset.
#%% md
## Data Extraction and Loading

Post-authorization, I elected to locally store the ~120 GB dataset in `PostgreSQL` over cloud-based access. This incurred a restricted download (500 kb/s) over ~19 hours for all zipped tables to complete. The end result, two datasets, comprising over half a million records, for over a quarter-million individuals.


| Records (Qty) | Scope              |
|---------------|--------------------|
| 364,627       | unique individuals |
| 546,028       | hospitalizations   |
| 94,458        | unique ICU stays   |


> `hosp` contains `546,028 hospitalizations` for `223,452 unique individuals`  
> `icu` contains `94,458 ICU stays` for `65,366 unique individuals`
#%% md
## Data Visual Inspection
With records unzipped and imported into `PostgreSQL` I could begin inspecting table columns.

## .High frequency of `null` values
Despite broad use, `null` values should remain in most cases.  
Larger tables combine various events that are inherently unique, and could be degraded in quality if subjected to rounding, interpolation or similar data manipulation. While handling varies by each case, generally speaking, high-level analysis may be inclined to `drop` such columns, whereas fine-grained analysis may `filter` records on specific data types and values.

## .Depersonalization; Modification of Date and Age values
Outlined in the official documentation, all personally identifiable information has been scrubbed from `MIMIC-IV`, and date/age values have been shifted at random, but retain their relation. These transformed values map to subsequent `anchor` columns, explained below:

The `anchor_year` column is a deidentified year occurring sometime between 2100 - 2200.  
The `anchor_year_group` column is one of the following values: "2008 - 2010", "2011 - 2013", "2014 - 2016", "2017 - 2019", and "2020 - 2022".
> Example: if a patient's `anchor_year` is 2158, and their `anchor_year_group` is 2011 - 2013, then any hospitalizations for the patient occurring in the year 2158 actually occurred sometime between 2011 - 2013.


The `anchor_age` provides the patient age in the given `anchor_year`.
> Example: If the patient was over 89 in the `anchor_year`, this `anchor_age` has been set to 91 (i.e. all patients over 89 have been grouped together into a single group with value 91, regardless of what their real age was).

## .Summary
To reiterate, retaining `null` values will vary by context. From basic analysis, it appears they will be ignored/dropped when performing broad-spectrum queries, where such detailed values would be irrelevant. Conversely, specific analysis, such as queries based on a specific condition, medication, or person, may benefit from retaining them, providing detailed insight on both condition and treatment.

This review affirms the machine learning algorithms and models chosen in the project proposal. There is a substantial amount of variability, with highly dimensional tables spanning a very broad range of topics and events. Selection of Random Forest, Gradient Descent and Neural Networks is far more applicable than their linear counterparts, which would struggle with overfitting and context development.