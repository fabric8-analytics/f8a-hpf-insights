"""Test the flask endpoints with local data."""

import unittest
import json

import src.flask_endpoint as rest_api


class TestFlaskMethods(unittest.TestCase):
    """Test the flask endpoints with local data."""

    def setUp(self):
        """Create a api client."""
        rest_api.app.app.testing = True
        self.client = rest_api.app.app.test_client()
        assert self.client is not None

    def test_health_checks(self):
        """Test the liveness and readiness probes."""
        self.assertEqual(self.client.get('/api/v1/liveness').status, '200 OK')
        self.assertEqual(self.client.get(
            '/api/v1/readiness').status, '200 OK')

    def test_root_path(self):
        """Test the root path."""
        self.assertEqual(self.client.get('/').status, '200 OK')

    def test_model_details(self):
        """Test model details endpoint."""
        self.assertEqual(self.client.get(
            '/api/v1/model_details').status, '200 OK')

    def test_companion_recommendations(self):
        """Test companion recommendations endpoint."""
        data = [{"ecosystem": "maven",
                 "package_list": ["org.springframework.cloud:spring-cloud-spring-service-connector"]
                 }]
        self.assertEqual(self.client.post(
            '/api/v1/companion_recommendation',
            data=json.dumps(data),
            headers={'content-type': 'application/json'}).status, '200 OK')


if __name__ == '__main__':
    unittest.main()
