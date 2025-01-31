# Summary of the Code:
This Python script analyzes the investment in solar photovoltaic (PV) systems under two different economic scenarios: standalone economy and sharing economy. The primary goal is to determine optimal investment strategies in terms of the area of PV installed by households and the profits earned, considering operational costs, net metering, and the impact of natural disasters. The script utilizes real-world energy consumption and solar generation data to perform computational experiments.

# Key Components of the Code:
# Data Preprocessing:

1. Loads a dataset containing hourly energy usage and solar generation data for different households from a CSV file. 2.Extracts valid data IDs and constructs matrices for:
   a. Solar generation per kW for each household.
   b. Energy consumption (load) per household.

# Investment Cost Calculation:

1. Computes the annuity cost of PV systems based on a discount rate (5%) and a lifespan of 20 years.
2. Defines key pricing parameters:
  a. Retail electricity price (pi_r)
  b. Net metering price (pi_nm)
  c. Investment cost per time step (pi_s)
# Investment Capacity Determination:

1. Computes the maximum possible PV investment for each household based on their annual energy consumption and generation capacity.
2. Ensures that households do not become net producers.

# Solving the Sharing Economy Scenario:

1. Implements a sharing collective investment model, where a subset of households jointly invest in PV to optimize overall benefits.
2. Uses a heuristic algorithm to:
  a. Sort firms based on potential generation capacity.
  b. Divide them into investors and non-investors.
  c. Iteratively adjust the investment set based on profitability criteria.
3. Computes the total PV investment in the sharing scenario.

# Solving the Standalone Economy Scenario:

1. Each household independently decides on PV investment to minimize energy costs.
2. Uses a numerical optimization approach to determine the optimal PV investment for each household.
3. Computes the total PV investment in the standalone scenario.

# Comparative Analysis of Investment Strategies:

1. Computes and plots key comparisons:
2. Investment differences between sharing and standalone cases.
3. Impact of operational price variations on investment.
4 .Effect of varying natural disaster probabilities (lambda) on PV investment.
5. Profitability of PV investment under different scenarios.
6. Outputs final investment decisions and their respective profitability metrics.

# Graphical Representation:

1. Plots:
  a. Total PV investment under different conditions.
  b. Profitability of PV investments in standalone vs. sharing economies.
  c. Investment patterns across different households.
  d. Impact of operational price and natural disasters on investment behavior.

