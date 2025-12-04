from locust import HttpUser, task, between

class User(HttpUser):
    wait_time = between(1, 3)  # Simulates a wait time between requests

    @task
    def get_home(self):
        self.client.get("/")
