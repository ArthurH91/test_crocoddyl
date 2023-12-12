import numpy as np
import matplotlib.pyplot as plt

# Define the function
def my_function(d):
    return 1 - d**2

# Generate x values
d_values = np.linspace(-2, 2, 100)

# Calculate corresponding y values
y_values = my_function(d_values)

# Plot the function
plt.plot(d_values, y_values, label='d^2 - 1')
plt.axhline(0, color='black', linewidth=0.5, linestyle='--', label='y = 0')  # Add a horizontal line at y=0
plt.axvline(0, color='black', linewidth=0.5, linestyle='--')  # Add a vertical line at x=0

# Add labels and title
plt.xlabel('d')
plt.ylabel('y')
plt.title('Plot of the function $1 - d^2 $')
plt.legend()  # Display legend

# Show the plot
plt.grid(True)
plt.show()