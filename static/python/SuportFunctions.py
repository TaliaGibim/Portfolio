# %% Max Polling

import numpy as np
import pandas as pd


def average_pooling(X, pool_size, stride):
    """
    Applies average pooling to a 2D input array.
    
    Parameters:
      X         : 2D NumPy array (for example, a grayscale image or a feature map).
      pool_size : tuple specifying the height and width of the pooling window.
      stride    : tuple specifying the vertical and horizontal stride.
      
    Returns:
      pooled    : 2D NumPy array after applying average pooling.
    """
    # Dimensions of the input array
    h, w = X.shape
    ph, pw = pool_size
    sh, sw = stride
    
    # Calculate the dimensions of the output array
    out_h = (h - ph) // sh + 1
    out_w = (w - pw) // sw + 1
    pooled = np.zeros((out_h, out_w))
    
    # Slide the pooling window across the input array
    for i in range(out_h):
        for j in range(out_w):
            start_i = i * sh
            start_j = j * sw
            window = X[start_i:start_i+ph, start_j:start_j+pw]
            pooled[i, j] = np.mean(window)
    
    return pooled

def anovaTest():
    # Defining the accuracy data for each model
    model1 = [0.10, 0.22, 0.47, 0.40, 0.63, 0.71, 0.56, 0.70, 0.70, 0.10, 0.19, 0.66, 0.60, 0.79, 0.88, 0.75, 0.82, 0.88]
    model2 = [0.82, 0.84, 0.86, 0.86, 0.88, 0.89, 0.81, 0.88, 0.90, 0.81, 0.84, 0.86, 0.86, 0.88, 0.89, 0.79, 0.89, 0.90]
    model3 = [0.14, 0.14, 0.18, 0.16, 0.22, 0.15, 0.21, 0.21, 0.16, 0.10, 0.11, 0.21, 0.20, 0.23, 0.22, 0.23, 0.22, 0.14]

    # Combine data into a DataFrame for convenience
    data = pd.DataFrame({'Model1': model1, 'Model2': model2, 'Model3': model3})
    # Kolmogorov-Smirnov test to check normality for each model
    def ks_test(models):
        results = {}
        for i, model in enumerate(models, 1):
            stat, p = ks_2samp(model, np.random.normal(np.mean(model), np.std(model), 1000))
            results[f'Model {i}'] = {'KS Statistic': stat, 'p-value': p}
        return results

    # Run KS test on the models
    normality_results = ks_test([model1, model2, model3])
    print("\nKolmogorov-Smirnov Test Results (Normality Check):")
    for model, result in normality_results.items():
        print(f"{model}: KS Stat = {result['KS Statistic']:.4f}, p-value = {result['p-value']:.4f}")
    # Perform One-Way ANOVA
    f_stat, p_value = f_oneway(model1, model2, model3)

    print(f"\nANOVA Test Statistic: {f_stat:.4f}, p-value = {p_value:.4f}")
    if p_value < 0.05:
        # Combine all models into a single array and their labels
        data_combined = np.concatenate([model1, model2, model3])
        labels = ['Model 1'] * len(model1) + ['Model 2'] * len(model2) + ['Model 3'] * len(model3)

        # Perform Tukey's HSD test
        tukey_result = pairwise_tukeyhsd(data_combined, labels)
        print("\nTukey's HSD Test Results (Post-hoc Pairwise Comparisons):")
        print(tukey_result)
    else:
        print("No significant difference found with ANOVA. Tukey's HSD skipped.")
    # Calculate Effect Size (η²)
    def calculate_eta_squared(f_stat, N):
        return f_stat / (N - 1)

    # Calculate the total number of observations
    N = len(model1) + len(model2) + len(model3)

    # Calculate η²
    eta_squared = calculate_eta_squared(f_stat, N)
    print(f"\nEffect Size (η²): {eta_squared:.4f}")


