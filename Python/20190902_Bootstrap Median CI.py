# Import packages
import numpy
import random
import pandas
import scipy.stats
import math
from dataclasses import dataclass
import bootstrapped.bootstrap as bs
import bootstrapped.stats_functions as bs_stats

# Establish random seed
numpy.random.seed(123)
random.seed(123)

"""
Create function to empirically derive confidence interval for the median of a numerical vector
"""
# Create n number of resamples in a Matrix
@dataclass
class ConfidenceInterval:
    sample_median: float
    upperlimit: float
    lowerlimit: float
    n: int
    confint: float


def bootstrap_median_ci(
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

    sample_median = numpy.median(sample)  # Store the sample median

    # Take the median of the row and subtract by sample median to get the variance
    variance = numpy.median(
        matrix
        , axis=1  # row axis
    ) - sample_median

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
    upperlimit = sample_median - ul
    lowerlimit = sample_median - ll

    return ConfidenceInterval(
        sample_median
        , upperlimit
        , lowerlimit
        , n
        , confint
    )


"""
Testing Section
"""

# Store a normal distribution to use in comparing bootstrap_median_ci
normal_sample = numpy.random.normal(
    loc=0
    , scale=1
    , size=1000
)

# Create TEST object
TEST = bootstrap_median_ci(
    sample=normal_sample
    , n=5000
    , confint=0.95
)

# Evaluate if sample medians are identical
print(TEST.sample_median == numpy.median(normal_sample))

# Evaluate if the upper limits and lower limits are close to confidence interval assuming a normal distribution
normal_sample = numpy.sort(normal_sample)  # Sort the vector low to high
mid = ((len(normal_sample) + 1) / 2) - 1  # Store the midpoint estimate, subtract 1 for Python indexing
std_err = scipy.stats.norm.ppf(q=0.975) * math.sqrt(
    len(normal_sample) * 0.5 * (1 - 0.5))  # Store standard error of a binomial
sample_lower = normal_sample[int(numpy.floor(mid - std_err))]
sample_upper = normal_sample[int(numpy.ceil(mid + std_err))]

print(sample_lower / TEST.lowerlimit)  # Compare lowerlimit of normal CI vs bootstrap CI
print(sample_upper / TEST.upperlimit)  # Compare upperlimit of normal CI vs bootstrap CI

# Store the median confidence interval locations derived from binomial distribution
lower, upper = scipy.stats.binom.interval(
    0.95
    , normal_sample.shape[0]
    , 0.5
    , loc=0
)
lower -= 1  # Adjust the low estimate for Python indexing

print( sample_lower / normal_sample[int(lower)])  # Compare lower hand derived CI vs binomial distribution CI
print( sample_upper / normal_sample[int(upper)])  # Compare upper hand derived CI vs binomial distribution CI

# Store results from actual bootstrapped package
bootstrap_results = bs.bootstrap(
    values=normal_sample
    , stat_func=bs_stats.median
    , alpha=0.05
    , num_iterations=5000
)

print( sample_lower / bootstrap_results.lower_bound)  # Compare lower binomial distribution CI vs bootstrapped CI
print( sample_upper / bootstrap_results.upper_bound)  # Compare upper binomial distribution CI vs bootstrapped CI