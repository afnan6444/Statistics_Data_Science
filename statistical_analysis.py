# week7_statistics_analysis.py

# ==============================
# 📦 Import Required Libraries
# ==============================
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import statsmodels.api as sm

# ==============================
# 📂 Load Business Data
# ==============================
df = pd.read_csv("business_data.csv")

# Preview dataset
print("Data Preview:\n", df.head())

# ==============================
# Day 1: Descriptive Statistics
# ==============================
print("\n--- Descriptive Statistics ---")
print(df[['Quantity','Price','Total_Sales']].describe())

sales_mean = df['Total_Sales'].mean()
sales_median = df['Total_Sales'].median()
sales_mode = df['Total_Sales'].mode()[0]
sales_std = df['Total_Sales'].std()

print(f"Sales Mean: {sales_mean:.2f}")
print(f"Sales Median: {sales_median:.2f}")
print(f"Sales Mode: {sales_mode:.2f}")
print(f"Sales Std Dev: {sales_std:.2f}")

# ==============================
# Day 2: Data Distribution Analysis
# ==============================
plt.figure(figsize=(8,5))
sns.histplot(df['Total_Sales'], kde=True, bins=20)
plt.title("Total Sales Distribution")
plt.show()

# Normality Test (Shapiro-Wilk)
shapiro_test = stats.shapiro(df['Total_Sales'])
print("\nShapiro-Wilk Test:", shapiro_test)

# ==============================
# Day 3: Correlation Analysis
# ==============================
corr_matrix = df[['Quantity','Price','Total_Sales']].corr()
print("\nCorrelation Matrix:\n", corr_matrix)

plt.figure(figsize=(6,5))
sns.heatmap(corr_matrix, annot=True, cmap="coolwarm")
plt.title("Correlation Heatmap")
plt.show()

# ==============================
# Day 4: Hypothesis Testing
# ==============================
# Example: Compare Total Sales between two regions
north_sales = df[df['Region']=="North"]['Total_Sales']
south_sales = df[df['Region']=="South"]['Total_Sales']

t_stat, p_val = stats.ttest_ind(north_sales, south_sales, equal_var=False)
print(f"\nT-test (North vs South Sales): t={t_stat:.3f}, p={p_val:.4f}")

# ANOVA Example: Compare Sales across multiple regions
anova_result = stats.f_oneway(
    df[df['Region']=="North"]['Total_Sales'],
    df[df['Region']=="South"]['Total_Sales'],
    df[df['Region']=="East"]['Total_Sales']
)
print("ANOVA Result:", anova_result)

# ==============================
# Day 5: Confidence Intervals
# ==============================
confidence_level = 0.95
sales_mean = np.mean(df['Total_Sales'])
sales_sem = stats.sem(df['Total_Sales'])
ci = stats.t.interval(confidence_level, len(df['Total_Sales'])-1, loc=sales_mean, scale=sales_sem)
print(f"\n95% Confidence Interval for Sales: {ci}")

# ==============================
# Day 6: Regression Analysis
# ==============================
# Linear Regression: Total_Sales ~ Price + Quantity
X = df[['Price','Quantity']]
y = df['Total_Sales']

X = sm.add_constant(X)  # add intercept
model = sm.OLS(y, X).fit()
print("\n--- Regression Summary ---")
print(model.summary())

# Plot regression line (Sales vs Price)
plt.figure(figsize=(8,5))
sns.regplot(x="Price", y="Total_Sales", data=df, ci=95, line_kws={"color":"red"})
plt.title("Regression: Total Sales vs Price")
plt.show()

# ==============================
# Day 7: Business Insights
# ==============================
print("\n📊 Business Insights:")
print(f"Average Sales: {sales_mean:.2f}")
print(f"Sales 95% CI: {ci}")
print(f"Correlation (Sales-Price): {corr_matrix.loc['Total_Sales','Price']:.2f}")
print(f"T-test p-value (North vs South): {p_val:.4f}")
print(f"Regression R²: {model.rsquared:.2f}")
