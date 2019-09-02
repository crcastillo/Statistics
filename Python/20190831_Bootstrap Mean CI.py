# Import packages
import numpy
import random
import pandas
import scipy.stats
import math
from dataclasses import dataclass

# Establish random seed
numpy.random.seed(123)
random.seed(123)

"""
Create function to empirically derive confidence interval for the mean of a numerical vector
"""
# Create n number of resamples in a Matrix
@dataclass
class ConfidenceInterval:
    sample_mean: float
    upperlimit: float
    lowerlimit: float
    n: int
    confint: float

def bootstrap_mean_ci(
        sample  # a vector of the numbers to resample from
        , confint=0.95  # decimal (0,1)
        , n=1000
):
    matrix = []  # Instantiate the matrix to fill
    for i in range(1, n):  # Iteratively append rows of resampled vectors (with replacement)
        row = random.choices(
            population=sample
            , k=len(sample)
        )
        matrix.append(row)
    matrix = numpy.asmatrix(matrix)  # Ensure list is returned as actual matrix

    sample_mean = numpy.mean(sample)  # Store the sample mean

    # Take the mean of the row and subtract by sample mean to get the variance
    variance = numpy.mean(
        matrix
        , axis=1  # row axis
    ) - sample_mean

    variance = numpy.ndarray.tolist(variance)  # Convert to an array
    variance.sort()  # Sort variance ascending

    ul = numpy.quantile(  # Store the upper limit quantile
        variance
        , q=round((1 - confint) / 2, 16)
        , interpolation='nearest'
    )

    ll = numpy.quantile(  # Store the lower limit quantile
        variance
        , q=round(1 - (1 - confint) / 2, 16)
        , interpolation='nearest'
    )

    # Create the upper and lower limits of the confidence interval
    upperlimit = sample_mean - ul
    lowerlimit = sample_mean - ll

    return ConfidenceInterval(
        sample_mean
        , upperlimit
        , lowerlimit
        , n
        , confint
    )


"""
Testing Section
"""

# Store a normal distribution to use in comparing bootstrap_mean_ci
normal_sample = numpy.random.normal(
    loc=0
    , scale=1
    , size=100
)

# Create TEST object
TEST = bootstrap_mean_ci(
    sample=normal_sample
    , n=1000
    , confint=0.95
)

# Evaluate if sample means are identical
print(TEST.sample_mean==numpy.mean(normal_sample))

# Evaluate if the upper limits and lower limits are close to confidence interval assuming a normal distribution
std_err = scipy.stats.norm.ppf(q=0.975) * (normal_sample.std()/(len(normal_sample)**0.5))  # Create standard error

print((numpy.mean(normal_sample) - std_err) / TEST.lowerlimit)  # Compare lowerlimit of normal CI vs bootstrap CI
print((numpy.mean(normal_sample) + std_err) / TEST.upperlimit)  # Compare upperlimit of normal CI vs bootstrap CI

# Show results from using t-distribution confidence interval
print(
    scipy.stats.t.interval(
        0.95
        , len(normal_sample) - 1
        , loc = numpy.mean(normal_sample)
        , scale = scipy.stats.sem(normal_sample)
    )
)