# Customer Persona Segmentation

## Project Overview

This project focuses on creating customer personas by analyzing user characteristics such as:
- Country
- Device (Android / iOS)
- Gender
- Age

The main goal is to:  
Transform raw customer data into meaningful segments and estimate expected revenue per persona.

### Objectives

- Aggregate customer data
- Create level-based personas
- Segment customers based on revenue
- Understand high-value vs low-value customer groups

## Key Concepts

#### Persona Creation

We combine multiple categorical variables into a single identifier:  

Example:

USA_ANDROID_FEMALE_24_30

This represents a customer persona.

#### Aggregation

We group data using:

```
df.groupby([...]).agg({"PRICE": "mean"})
```

This helps us understand:

“How much does each persona spend on average?”

#### Segmentation 

Customers are divided into segments using:

```
pd.qcut(df["PRICE"], 4, labels=["D", "C", "B", "A"])
```

| Segment | Meaning |
| :--- | :--- | 
| A | Highest value customers|
| B | Above average |
| C | Below average |
| D | Lowest value |


## Technologies Used
- Python 3
- Pandas
- NumPy
- Jupyter Notebook


## Why This Project Matters

This project demonstrates:

- Data aggregation skills
- Feature engineering
- Customer segmentation logic
- Business-oriented thinking
