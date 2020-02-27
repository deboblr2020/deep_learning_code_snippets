# Graphs and charts using Seaborn


```python
import matplotlib.pyplot as plt
import seaborn as sns
sns.set(style="ticks", color_codes=True)


# Initialize a 2x2 grid of facets using the tips dataset:
g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill")

# Pass additional keyword arguments to the mapped function:
import numpy as np 
bins = np.arange(0, 65, 5)
g = sns.FacetGrid(tips, col="time",  row="smoker")
g = g.map(plt.hist, "total_bill", bins=bins, color="r")

# pair plot
sns.pairplot(df.iloc[:,7:].select_dtypes(include=['number']).fillna(0))

# corr
corr = d2.corr()
sns.heatmap(corr,xticklabels=corr.columns.values,yticklabels=corr.columns.values)
corr['pred_ind'].abs().sort_values(ascending=False)[:12]

# Assign one of the variables to the color of the plot elements:
g = sns.FacetGrid(tips, col="time",  hue="smoker")
g = (g.map(plt.scatter, "total_bill", "tip", edgecolor="w").add_legend())

# Change the height and aspect ratio of each facet:
g = sns.FacetGrid(tips, col="day", height=4, aspect=.5)
g = g.map(plt.hist, "total_bill", bins=bins)

# crosstab and check for abrplot
pd.crosstab(d2['ten_band'],d2['pred_ind'],normalize='index').plot(kind='bar')


```




