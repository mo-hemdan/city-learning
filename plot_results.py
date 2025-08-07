#%%
import matplotlib.pyplot as plt


x = ['betweeness', '+road_type']
y = [0.9935241383053709, 0.7635667244385481]

plt.bar(x, y, width=0.3)
plt.xlabel("Features")
plt.ylabel('Mean Absolute Error (nlanes)')
# %%
