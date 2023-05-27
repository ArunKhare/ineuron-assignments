import pandas as pd
import numpy as np
import matplotlib_inline
import matplotlib.pyplot as plt

if __name__ =="__main__":


    data = pd.read_csv(r"C:\Users\arunk\Assignments\ASSESSMENT COMPLETION\Statistic_Assessment_24_05_2023\data.csv")
    shape=data.shape
    mean_before = np.mean(data[' Blood Pressure Before (mmHg)'])
    mean_after = np.mean(data[' Blood Pressure After (mmHg)'])

    print(f"mean_after : {mean_after}, mean_after : {mean_before}")
    # A. Measure dispersion

    data["range_before"] = data[' Blood Pressure Before (mmHg)'] - data[' Blood Pressure After (mmHg)']
    mean_absolute_deviation_before = sum(abs(data[' Blood Pressure Before (mmHg)'] - mean_before)) / shape[0]
    mean_absolute_deviation_after = sum(abs(data[' Blood Pressure After (mmHg)'] - mean_after)) / shape[0]

    sum_squared_deviation_before = sum((data[' Blood Pressure Before (mmHg)'] - mean_before)**2)
    sum_squared_deviation_after = sum((data[' Blood Pressure After (mmHg)'] - mean_after)**2)

    variance_before = sum_squared_deviation_before/(shape[0]-1)
    sd_before = np.sqrt(variance_before)

    variance_after = sum_squared_deviation_after/(shape[0]-1)
    sd_after = np.sqrt(variance_after)


    max_before = max(data[" Blood Pressure Before (mmHg)"])
    min_before = min(data[" Blood Pressure Before (mmHg)"])
    max_after = max(data[' Blood Pressure After (mmHg)'])
    min_after = min(data[" Blood Pressure After (mmHg)"])

    range_before = max_after-min_before
    range_after = max_after-min_after
    print(f'range_before : {range_before} , range_after :  {range_after}')
    print(f'mean absolute deviation before: {mean_absolute_deviation_before}, mean absolute deviation after: {mean_absolute_deviation_after} ')
    print(f"Variance_before: , {variance_before}, sd_bfore: {sd_before}, Variance_after {variance_after}, sd_after {sd_after}")

    """interpretation 
    The mean absolute deviation (MAD) is a measure of the average distance between each data point and the mean. In this case, the mean absolute deviation before is approximately 5.7118, and the mean absolute deviation after is approximately 5.9.

    The MAD indicates the average amount of deviation or spread in the data from the mean value. A higher MAD suggests a higher variability or dispersion in the data points. In this case, both before and after measurements have a similar MAD, indicating a similar level of dispersion in the data.

    The variance and standard deviation provide additional measures of the spread of the data. The variance before is approximately 43.5373, with a standard deviation of approximately 6.5983. The variance after is approximately 47.4448, with a standard deviation of approximately 6.8880.

    The variance quantifies the average squared deviation from the mean, while the standard deviation is the square root of the variance and provides a measure of the average deviation from the mean. In this case, both before and after measurements have similar variances and standard deviations, indicating a similar level of dispersion or spread in the data.

    Overall, these measures of dispersion provide insights into the spread or variability of the blood pressure measurements before and after.
    """
    # b. Calculate mean and 5% confidence interval and plot it in a graph
        # critical_value = 0.05/2 = 1- 0.025 = P_value = AUC = 0.975 = Z_Score = 1.96( from Z table)
        # lower_bound = mean-("cirtical value"* sd/sqrt(n))
        # upper_bound = mean+("cirtical value"* sd/sqrt(n))
    critical_value  = 1.96
    lower_bound_before = mean_before - (1.96 * sd_before/np.sqrt(shape[0]))
    upper_bound_before = mean_before + (1.96 * sd_before/np.sqrt(shape[0]))
    print(f"Lower_bound_before : {lower_bound_before} Upper_bound_before : {upper_bound_before}")

    lower_bound_after = mean_after - (1.96 * sd_after/np.sqrt(shape[0]))
    upper_bound_after = mean_after + (1.96 * sd_after/np.sqrt(shape[0]))
    print(f"Lower_bound_after : {lower_bound_after}, Upper_bound_after : {upper_bound_after}")



    plt.figure(figsize=(8, 6))
    plt.errorbar(0, mean_before, yerr=(upper_bound_before - mean_before), fmt='o', label='Mean', color='blue')
    plt.axhline(y=mean_before, color='blue', linestyle='--', label='Mean')
    plt.axhline(y=lower_bound_before, color ='green', linestyle='--',label='Lower Bound (5% CI)')
    plt.axhline(y=upper_bound_before, color ='green', linestyle='--',label='Upper Bound (5% CI)')
    plt.xticks(range(1, shape[0]+1), data['Patient ID'])
    plt.xlabel("Patient ID")
    plt.ylabel("Blood Pressure Before(mmHg)")
    plt.title("Mean and 5% Confidence Interval")
    plt.legend()
    plt.show
    # The formula for Pearson's correlation coefficient (r) is:
    # r = Σ((x - x̄) * (y - ȳ)) / sqrt(Σ(x - x̄)^2 * Σ(y - ȳ)^2)
    # To perform the significance test, we calculate the t-value using the formula:
    # t = r * np.sqrt((n - 2) / (1 - r^2))

    from scipy.stats import pearsonr
    r, p_value = pearsonr(data[' Blood Pressure Before (mmHg)'], data[' Blood Pressure After (mmHg)'])
    t_value = r * np.sqrt((shape[0] - 2) / (1 - r**2))
    print(f" correlation coefficient : {r} ,P_ value { p_value}")
    print(f"t_value : {t_value}" )

    """ interpretation 
    Based on the calculated correlation coefficient of 0.9779, it indicates a strong positive correlation between the "Blood Pressure Before" and "Blood Pressure After" measurements. 
    The p-value of 1.8097e-68 indicates that the correlation coefficient is statistically significant at the 1% level of significance. This means that the observed correlation is highly unlikely to occur by chance, providing evidence to reject the null hypothesis of no correlation.
    The calculated t-value of 46.3557 indicates the strength of the correlation and is used to assess the significance of the correlation coefficient. A large absolute t-value suggests a stronger correlation, and in this case, the large t-value supports the significant correlation observed.


    """

