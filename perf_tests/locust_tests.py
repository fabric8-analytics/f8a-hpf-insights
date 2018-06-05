"""Load Testing Companion Recommendation."""
import json
from locust import HttpLocust, TaskSet, events, task, web
from collections import Counter
import random

# Use this for devcluster.
# stats = {"host-distribution": Counter()}

input_stack_list = [
    {
        "ecosystem": "maven",
        "package_list": ["io.vertx:vertx-core", "io.vertx:vertx-web"]

    },
    {
        "ecosystem": "maven",
        "package_list": ["io.vertx:vertx-core"]
    },
]


def get_packages():
    """Generate random len input stack for load testing."""
    input_stack_len = len(input_stack_list)
    random.shuffle(input_stack_list)
    input_random_stack = random.sample(
        input_stack_list, random.randint(1, input_stack_len))
    if len(input_random_stack) % 4 == 0:
        input_random_stack.append({{
            "ecosystem": "pypi",
            "package_list": ["numpy"]

        }})
    return input_random_stack


class HPFInsightsBehaviour(TaskSet):
    """This class defines the user behaviours."""

    def on_start(self):
        """on_start is called when a Locust start before any task is scheduled."""
        pass

    @task
    def trigger_companion_recommendation_random_stack_len(self):
        """Simulate a stack analysis request."""
        response = self.client.post("/api/v1/companion_recommendation",
                                    data=json.dumps(get_packages()),
                                    headers={'Content-type': 'application/json'})

        # Use the below lines for devcluster where we take into consideration the host pod as well.
        # stats["host-distribution"][response.json()['HOSTNAME']] += 1
        # Use this for devcluster.
        # print(stats['host-distribution'])


class HPFInsightsLocust(HttpLocust):
    """This class defines the params for the load testing piece."""

    task_set = HPFInsightsBehaviour
    min_wait = 10
    max_wait = 10
