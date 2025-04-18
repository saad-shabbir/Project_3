# Market Scoring Simulation Tool

**Developed by**: Saad Shabbir and Cam Heitkamp  
**Course**: Machine Learning for Business Analysts with Professor Tom Mattson

## Project Overview

This Python script is an interactive simulation tool that helps analysts evaluate potential international markets for expansion. By integrating user-driven simulations and statistical modeling, the program analyzes market viability across 30 countries based on various macroeconomic and risk-based indicators.

## Dataset Description

The tool uses structured data stored in a JSON file named `Market_Scores.json`, which contains the following for each country:

### Main Features per Country:

- **Market Size**: Average and standard deviation  
- **Growth Rate**: Average and standard deviation  
- **Political & Economic Risk**: Average and standard deviation  
- **Talent Risk**: Minimum and maximum range  
- **Competition Risk**: Minimum and maximum range  

All numeric inputs are used in simulation logic using random number generation via Gaussian or uniform distributions depending on the feature.

## Simulation Methodology

### Step-by-step Logic:

1. **User Input**: The script prompts for the number of iterations (between 500 and 25,000) for simulation accuracy.  
2. **Simulation**: For each country, values are simulated across 5 risk/market dimensions using appropriate distributions (log-normal or uniform).  

### Scoring Formula:

```
Market Score = 10 + 0.22 × Size + 0.64 × Growth Rate 
             - 0.65 × Competition Risk 
             - 1.03 × Political/Economic Risk 
             - 1.37 × Talent Risk
```

3. **Statistical Analysis**: Calculates mean, standard deviation, median, min, max, and range for each country's scores.  
4. **Output Storage**: Saves simulation results (including raw scores) to a pipe-delimited text file named `Market_Simulation_Results.txt`.  
5. **Top/Bottom Ranking**: Asks the user to choose how many top and bottom results to display (1–10), then lets them explore rankings by metric.

## Output File

The output text file includes:

- Country Name  
- Average Market Score  
- Standard Deviation  
- Median  
- Minimum  
- Maximum  
- Range  
- Raw simulation results (list format)  

Each line is separated by the `|` pipe delimiter for easy parsing or import into Excel or other tools.

## User Decisions

- Number of simulation iterations (500 to 25,000)  
- Number of top/bottom countries to display (1–10)  
- Metric for ranking: average, standard deviation, median, min, max, or range  
- Option to rerun ranking with a different metric  

## Example Workflow

- User runs script and chooses 5,000 iterations  
- User chooses to display top and bottom 5 countries  
- User selects "Average" as the ranking metric  
- Script outputs rankings to terminal and saves full simulation to a text file  
- User enters "Done" when satisfied, ending the program  

## Requirements

- Python 3.x  
- `Market_Scores.json` in the same directory  

### Libraries Used:

- `math`  
- `statistics`  
- `csv`  
- `random`  
- `json`  

## Potential Extensions

- Add visualization (matplotlib/seaborn)  
- GUI for input and output display  
- Integrate confidence intervals or bootstrap analysis  
- Support Excel or SQL as input/output formats  

---

Run the final line of the script to begin the interactive simulation experience:

```python
run_simulation()
```
