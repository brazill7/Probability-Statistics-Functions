# Stat & Prob Functions
# Maverick Brazill

import numpy as np
import math
import scipy.stats as stats
import scipy.optimize as optimize
import matplotlib.pyplot as plt

def calculate_sample_mean(sample):
    """
    Calculate the mean of a sample.

    Parameters:
    - sample: The input sample.

    Returns:
    - mean: The mean of the sample.
    """
    if len(sample) == 0:
        raise ValueError("Sample size == 0")
    
    return sum(sample) / len(sample)

def calculate_sample_median(sample):
    """
    Calculate the median of a sample.

    Parameters:
    - sample: The input sample.

    Returns:
    - median: The median of the sample.
    """
    sorted_sample = sorted(sample)
    n = len(sorted_sample)

    if n % 2 == 0:
        middle_left = sorted_sample[n // 2 - 1]
        middle_right = sorted_sample[n // 2]
        median = (middle_left + middle_right) / 2
    else:
        median = sorted_sample[n // 2]

    return median

def calculate_sample_std_dev(sample):
    """
    Calculate the standard deviation of a sample.

    Parameters:
    - sample: The input sample.

    Returns:
    - std_dev: The standard deviation of the sample.
    """
    n = len(sample)

    if n < 2:
        raise ValueError("Sample size should be at least 2 for standard deviation calculation.")

    mean = sum(sample) / n
    squared_diff_sum = sum((x - mean) ** 2 for x in sample)
    variance = squared_diff_sum / (n - 1)  # Use (n-1) for sample standard deviation

    std_dev = math.sqrt(variance)
    return std_dev

def calculate_confidence_interval(sample, confidence_level=0.95):
    """
    Calculate a confidence interval for the mean of a sample.

    Parameters:
    - sample: The input sample.
    - confidence_level: The confidence level for the interval (default is 0.95).

    Returns:
    - confidence_interval: A tuple representing the confidence interval.
    """
    if len(sample) == 0:
        raise ValueError("Sample size == 0")

    n = len(sample)
    sample_mean = calculate_sample_mean(sample)
    standard_error = stats.sem(sample)
    margin_of_error = stats.t.ppf((1 + confidence_level) / 2, n - 1) * standard_error
    confidence_interval = (sample_mean - margin_of_error, sample_mean + margin_of_error)

    return confidence_interval

def proportion_atleast_above(sample, atleast):
    """
    Calculate the proportion of values in a sample that are at least a specified value.

    Parameters:
    - sample: The input sample.
    - atleast: The threshold value.

    Returns:
    - proportion: The proportion of values in the sample that are at least the specified value.
    """
    if len(sample) == 0:
        raise ValueError("Sample size == 0")

    count_above_num = sum(1 for level in sample if level >= atleast)
    proportion = count_above_num / len(sample)

    return proportion

def estimate_lognormal_parameters(sample):
    """
    Estimate the parameters (mu, sigma) of a log-normal distribution from a sample.

    Parameters:
    - sample: The input sample.

    Returns:
    - mu: The estimated mean parameter.
    - sigma: The estimated standard deviation parameter.
    """
    log_data = np.log(sample)
    mu = np.mean(log_data)
    sigma = np.std(log_data)

    return mu, sigma

def expected_value(sample):
    """
    Calculate the expected value of a sample.

    Parameters:
    - sample: The input sample.

    Returns:
    - expected_value: The expected value of the sample.
    """
    total_observations = len(sample)
    unique_values = set(sample)

    x_values = list(unique_values)
    probabilities = [sample.count(x) / total_observations for x in x_values]

    expected_value = sum(x * p for x, p in zip(x_values, probabilities))
    return expected_value

def binomial_mle(n, x):
    """
    Perform maximum likelihood estimation (MLE) for a binomial distribution.

    Parameters:
    - n: The number of trials.
    - x: The number of successes.

    Returns:
    - mle_p: The MLE estimate for the probability parameter.
    - bias: The bias of the estimate.
    """
    # Define the negative log-likelihood function
    def neg_log_likelihood(p):
        return - (np.log(comb(n, x)) + x * np.log(p) + (n - x) * np.log(1 - p))

    # Initial guess for the optimizer (p should be between 0 and 1)
    initial_guess = 0.5

    # Minimize the negative log-likelihood function
    result = optimize.minimize(neg_log_likelihood, initial_guess, bounds=[(0, 1)])

    # Extract the MLE for p
    mle_p = result.x[0]

    # Check for bias
    true_p = x / n  # True probability parameter for a binomial distribution
    bias = mle_p - true_p

    return mle_p, bias

def probability_negative_tests(n, x, num_tests):
    """
    Calculate the probability of having 0 successes (negative tests) in a binomial distribution.

    Parameters:
    - n: The number of trials.
    - x: The number of successes.
    - num_tests: The number of tests.

    Returns:
    - probability: The probability of having 0 successes in the specified number of tests.
    """
    # Probability of success in a single test
    p = x / n

    # Probability of having 0 successes (negative tests) in num_tests trials
    probability = comb(num_tests, 0) * p**0 * (1 - p)**num_tests

    return probability

def margin_of_error_z(confidence_level=0.95):
    """
    Calculate the margin of error for a Z-test.

    Parameters:
    - confidence_level: The confidence level (default is 0.95).

    Returns:
    - margin_of_error: The margin of error.
    """
    return stats.norm.ppf((1 + confidence_level) / 2)

def calculate_interval_when(sample_mean, sample_size, population_std, confidence_level=0.95):
    """
    Calculate the confidence interval for a population mean using a Z-test.

    Parameters:
    - sample_mean: The sample mean.
    - sample_size: The sample size.
    - population_std: The population standard deviation.
    - confidence_level: The confidence level for the interval (default is 0.95).

    Returns:
    - lower_bound: The lower bound of the confidence interval.
    - upper_bound: The upper bound of the confidence interval.
    """
    # Calculate the margin of error
    margin_of_error = margin_of_error_z(confidence_level) * (population_std / (sample_size ** 0.5))

    # Calculate the confidence interval
    lower_bound = sample_mean - margin_of_error
    upper_bound = sample_mean + margin_of_error

    return lower_bound, upper_bound

def create_box_plot(data, labels=None):
    """
    Create a box plot.

    Parameters:
    - data: The input data for the box plot.
    - labels: Labels for the box plot categories (default is None).
    """
    plt.boxplot(data, labels=labels, vert=False)
    plt.title('Box Plot')
    plt.xlabel('Values')
    plt.ylabel('Categories')
    plt.show()

def t_critical_value_test(confidence_level, df, two_tailed=True):
    """
    Calculate the critical t-value for a t-test.

    Parameters:
    - confidence_level: The confidence level.
    - df: Degrees of freedom.
    - two_tailed: Whether the test is two-tailed (default is True).

    Returns:
    - t_critical: The critical t-value.
    """
    alpha_over_2 = confidence_level / 2 if two_tailed else confidence_level

    t_critical = t.ppf(1 - alpha_over_2, df)

    return t_critical

def confidence_intervals_for_variance_and_stddev(n, s, confidence_level=0.95):
    """
    Calculate confidence intervals for population variance and standard deviation.

    Parameters:
    - n: Sample size.
    - s: Sample standard deviation.
    - confidence_level: The confidence level for the intervals (default is 0.95).

    Returns:
    - var_ci: Confidence interval for population variance.
    - std_ci: Confidence interval for population standard deviation.
    """
    # Degrees of freedom
    df = n - 1

    # Confidence interval for the population variance
    lower_var = (n - 1) * s**2 / stats.chi2.ppf(1 - confidence_level / 2, df)
    upper_var = (n - 1) * s**2 / stats.chi2.ppf(confidence_level / 2, df)

    # Confidence interval for the population standard deviation
    lower_std = math.sqrt(lower_var)
    upper_std = math.sqrt(upper_var)

    var_ci = (lower_var, upper_var)
    std_ci = (lower_std, upper_std)

    return var_ci, std_ci

def hypothesis_test(p_value, alpha):
    """
    Perform a hypothesis test based on a given p-value and significance level.

    Parameters:
    - p_value: The p-value.
    - alpha: The significance level.

    Returns:
    - result: A string indicating the outcome of the hypothesis test.
    """
    if p_value <= alpha:
        result = f'Reject H0: p-value ({p_value:.4f}) <= alpha ({alpha})'
    else:
        result = f'Fail to reject H0: p-value ({p_value:.4f}) > alpha ({alpha})'

    return result

def z_test(sample_mean, population_mean_under_null, population_std, sample_size, significance_level):
    """
    Perform a Z-test and determine whether to reject the null hypothesis.

    Parameters:
    - sample_mean: The sample mean.
    - population_mean_under_null: The population mean under the null hypothesis.
    - population_std: The population standard deviation.
    - sample_size: The sample size.
    - significance_level: The significance level for the test.

    Returns:
    - result: A dictionary containing the Z-test statistic, p-value, and rejection decision.
    """
    # Calculate Z-test statistic
    z_stat = (sample_mean - population_mean_under_null) / (population_std / (sample_size ** 0.5))

    # Calculate p-value
    p_value = 1 - stats.norm.cdf(z_stat)

    # Check if we should reject the null hypothesis
    reject_null = p_value < significance_level

    # Output the results
    result = {
        'Z-test Statistic': z_stat,
        'P-value': p_value,
        'Reject Null Hypothesis': reject_null
    }

    return result

def calculate_type_ii_error_probability(population_mean_under_null, population_mean_alternative, population_std, sample_size, significance_level):
    """
    Calculate the probability of Type II error for a one-sample Z-test.

    Parameters:
    - population_mean_under_null: The population mean under the null hypothesis.
    - population_mean_alternative: The true population mean under the alternative hypothesis.
    - population_std: The population standard deviation.
    - sample_size: The sample size.
    - significance_level: The significance level for the test.

    Returns:
    - beta: The probability of Type II error.
    """
    # Calculate the critical value for the alternative hypothesis
    critical_value = stats.norm.ppf(1 - significance_level)

    # Calculate the standard error
    standard_error = population_std / (sample_size ** 0.5)

    # Calculate the alternative hypothesis mean under the null hypothesis
    mu_alternative_under_null = population_mean_under_null + critical_value * standard_error

    # Calculate the probability of Type II error (beta)
    beta = stats.norm.cdf(mu_alternative_under_null, loc=population_mean_alternative, scale=standard_error)

    return beta

def calculate_sample_size_for_type_ii_error(population_mean_under_null, population_mean_alternative, population_std, significance_level, desired_type_ii_error):
    """
    Calculate the necessary sample size to achieve a specific Type II error rate.

    Parameters:
    - population_mean_under_null: The population mean under the null hypothesis.
    - population_mean_alternative: The true population mean under the alternative hypothesis.
    - population_std: The population standard deviation.
    - significance_level: The significance level for the test.
    - desired_type_ii_error: The desired Type II error rate.

    Returns:
    - sample_size: The necessary sample size.
    """
    # Calculate critical values for the significance level and desired Type II error rate
    z_alpha = stats.norm.ppf(1 - significance_level / 2)
    z_beta = stats.norm.ppf(desired_type_ii_error)

    # Calculate the noncentrality parameter (delta)
    delta = population_mean_alternative - population_mean_under_null

    # Calculate the necessary sample size
    sample_size = ((z_alpha - z_beta) * population_std / delta) ** 2

    return math.ceil(sample_size)  # Round up to ensure an integer sample size

def calculate_p_value(z_statistic, alternative_hypothesis='two-sided'):
    """
    Calculate the p-value for a one-sample proportion test.

    Parameters:
    - z_statistic: The Z-test statistic.
    - alternative_hypothesis: The alternative hypothesis. Options: 'two-sided', 'greater', 'less'.
                              Default is 'two-sided'.

    Returns:
    - p_value: The p-value.
    """
    if alternative_hypothesis == 'two-sided':
        p_value = 2 * stats.norm.sf(abs(z_statistic))
    elif alternative_hypothesis == 'greater':
        p_value = stats.norm.sf(z_statistic)
    elif alternative_hypothesis == 'less':
        p_value = stats.norm.cdf(z_statistic)
    else:
        raise ValueError("Invalid alternative_hypothesis. Choose from 'two-sided', 'greater', 'less'.")

    return p_value

def two_tailed_z_test(sample_mean, population_mean, population_std, sample_size, significance_level):
    """
    Perform a two-tailed Z-test and determine whether to reject the null hypothesis.
    Parameters:
    - sample_mean: The sample mean.
    - population_mean: The population mean under the null hypothesis.
    - population_std: The population standard deviation.
    - sample_size: The sample size.
    - significance_level: The significance level for the test.

    Returns:
    - result: A dictionary containing the Z-test statistic, critical values, and rejection decision.
    """
    # Calculate the Z-test statistic
    z_stat = (sample_mean - population_mean) / (population_std / (sample_size ** 0.5))

    # Find the critical values for the two-tailed test
    critical_value_lower = stats.norm.ppf(significance_level / 2)
    critical_value_upper = stats.norm.ppf(1 - significance_level / 2)

    # Check if the Z-test statistic is outside the critical region
    reject_null = z_stat < critical_value_lower or z_stat > critical_value_upper

    # Output the results
    result = {
        'Z-test Statistic': z_stat,
        'Critical Value Lower': critical_value_lower,
        'Critical Value Upper': critical_value_upper,
        'Reject Null Hypothesis': reject_null
    }

    return result

def sample_t_test(sample_size, t_stat, alpha):
    """
    Perform a one-sample t-test and make a conclusion.

    Parameters:
    - sample_size: The sample size.
    - t_stat: The t-statistic.
    - alpha: The significance level for the test.

    Returns:
    - conclusion: A string indicating the conclusion of the hypothesis test.
    """
    # Degrees of freedom
    df = sample_size - 1

    # Calculate the p-value
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df))

    # Make a conclusion based on the p-value and significance level
    if p_value < alpha:
        conclusion = f"Reject H0: p-value ({p_value:.4f}) < alpha ({alpha})"
    else:
        conclusion = f"Fail to reject H0: p-value ({p_value:.4f}) >= alpha ({alpha})"

    return conclusion

def calculate_standard_error(sample_std, sample_size):
    """
    Calculate the standard error of the mean.

    Parameters:
    - sample_std: The sample standard deviation.
    - sample_size: The sample size.

    Returns:
    - standard_error: The standard error of the mean.
    """
    return sample_std / math.sqrt(sample_size)

def run_calculations():
    print("enter stuff here")

if __name__ == "__main__":
    run_calculations()

