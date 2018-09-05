"""Load Testing Companion Recommendation."""

import json
from locust import HttpLocust, TaskSet, task
from collections import Counter
import random

stats = {"host-distribution": Counter()}

input_stack_list = [
    {
        "ecosystem": "maven",
        "package_list": ["io.vertx:vertx-core", "io.vertx:vertx-web"]

    },
    {
        "ecosystem": "maven",
        "package_list": ["io.vertx:vertx-core"]
    },
    {
        "ecosystem": "maven",
        "package_list": ["org.sakaiproject.velocity:sakai-velocity-tool",
                         "org.sakaiproject:sakai-rights-api",
                         "org.sakaiproject.kernel:sakai-kernel-api",
                         "org.sakaiproject.kernel:sakai-kernel-util",
                         "org.sakaiproject.courier:sakai-courier-api",
                         "org.sakaiproject.message:sakai-message-api",
                         "commons-logging:commons-logging",
                         "org.sakaiproject.site:sakai-mergedlist-util",
                         "org.sakaiproject.presence:sakai-presence-util",
                         "javax.servlet:servlet-api",
                         "org.sakaiproject.presence:sakai-presence-api",
                         "org.sakaiproject.courier:sakai-courier-util",
                         "org.sakaiproject.kernel:sakai-component-manager",
                         "org.sakaiproject.velocity:sakai-velocity-tool-api"]
    },
    {
        "ecosystem": "maven",
        "package_list": ["org.webjars:bootstrap",
                         "org.springframework.boot:spring-boot-starter-cache",
                         "org.springframework.boot:spring-boot-starter-thymeleaf",
                         "org.springframework.session:spring-session",
                         "com.h2database:h2",
                         "org.springframework.boot:spring-boot-starter-data-jpa",
                         "org.springframework.boot:spring-boot-devtools",
                         "org.springframework.boot:spring-boot-starter-web",
                         "mysql:mysql-connector-java",
                         "org.thymeleaf.extras:thymeleaf-extras-springsecurity4"]},
    {"ecosystem": "maven",
     "package_list": ["org.kitesdk:kite-data-hive",
                      "org.slf4j:slf4j-api",
                      "org.kitesdk:kite-data-core",
                      "org.apache.hive:hive-common",
                      "org.apache.hive:hive-service",
                      "org.apache.hive:hive-metastore",
                      "org.apache.hive:hive-exec",
                      "org.apache.flume:flume-ng-node",
                      "org.kitesdk:kite-hbase-cdh5-test-dependencies",
                      "org.kitesdk:kite-hadoop-cdh5-dependencies",
                      "com.beust:jcommander",
                      "org.kitesdk:kite-hbase-cdh5-dependencies",
                      "org.apache.hive:hive-serde",
                      "org.kitesdk:kite-hadoop-cdh5-test-dependencies"]},
    {"ecosystem": "maven",
     "package_list": ["org.slf4j:slf4j-api",
                      "org.apache.commons:commons-lang3",
                      "com.fasterxml.jackson.core:jackson-annotations",
                      "com.fasterxml.jackson.core:jackson-core",
                      "org.apache.calcite.avatica:avatica-core",
                      "org.apache.calcite:calcite-linq4j",
                      "commons-dbcp:commons-dbcp",
                      "com.google.code.findbugs:jsr305",
                      "org.codehaus.janino:janino",
                      "com.fasterxml.jackson.core:jackson-databind",
                      "org.codehaus.janino:commons-compiler",
                      "com.esri.geometry:esri-geometry-api",
                      "com.google.guava:guava",
                      "net.hydromatic:aggdesigner-algorithm"]}
]


def get_packages():
    """Generate random len input stack for load testing."""
    input_stack_len = len(input_stack_list) - 1
    random.shuffle(input_stack_list)
    return [input_stack_list[random.randint(0, input_stack_len)]]


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
                                    headers={"Content-type": "application/json"})
        assert response


class HPFInsightsLocust(HttpLocust):
    """This class defines the params for the load testing piece."""

    task_set = HPFInsightsBehaviour
    min_wait = 10
    max_wait = 10
