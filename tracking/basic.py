PROJECT_NAME = "quick-start"

# Make sure a project exists
from polyaxon.client import ProjectClient

project = ProjectClient(project=PROJECT_NAME)
project.get_or_create()


# Init a new run
from traceml import tracking

tracking.init(project="quick-start")

# Tracking experiment
import datetime
import random
from time import sleep

# log inputs/parameters
tracking.log_inputs(param1=0.21, param2="foo", param3=False)

# log outputs/results
tracking.log_outputs(val1=0.23, val2="bar", val3=True)

# log metrics (flavor one): single metric at a time
for i in range(100):
    sleep(0.2)
    # Timeseries with no step logic, timestamp is automatically tracked
    tracking.log_metric(name="single_metric1", value=random.random())
    # Timeseries with step-wise logic, timestamp is automatically tracked
    tracking.log_metric(name="single_metric2", value=i * random.random(), step=i)
    # Timeseries with step-wise logic, timestamp is tracked manually as a user input
    tracking.log_metric(
        name="single_metric3",
        value=(1 / i) * random.random(),
        step=i,
        timestamp=datetime.datetime.now() + datetime.timedelta(seconds=i * 4),
    )

# log metrics (flavor 2): multiple metrics that share similar tracking logic (step and timestamp)
for i in range(100):
    sleep(0.2)
    # Multiple metrics tracked in a single call with no step logic,
    # timestamp is automatically tracked
    tracking.log_metrics(
        multi_metric11=random.random(),
        multi_metric12=i * random.random(),
        multi_metric13=(1 / i) * random.random(),
    )
    # Multiple metrics tracked in a single call with step-wise logic,
    # timestamp is automatically tracked
    tracking.log_metrics(
        multi_metric21=random.random(),
        multi_metric22=i * random.random(),
        multi_metric23=(1 / i) * random.random(),
        step=i,
    )
    # Multiple metrics tracked in a single call with step-wise logic,
    # timestamp is tracked manually as a user input
    tracking.log_metrics(
        multi_metric31=random.random(),
        multi_metric32=i * random.random(),
        multi_metric33=(1 / i) * random.random(),
        step=i,
        timestamp=datetime.datetime.now() + datetime.timedelta(seconds=i * 4),
    )
