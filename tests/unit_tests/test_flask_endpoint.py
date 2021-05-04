"""Test the flask endpoints with local data."""

import json
import mock
import pytest
import random
from faker import Faker

faker = Faker()


class MockHPFScoring(mock.Mock):
    """Mock HPFScoring Class."""

    def __init__(self, *args, **kwargs):
        """Initialize the object."""
        super().__init__(*args, **kwargs)

    def predict(self, input_stack):
        """Predict the stacks."""
        pkg_id_list, missing_pkg = self._map_input_to_package_ids(input_stack)
        if len(pkg_id_list) < len(missing_pkg):
            return [], list(missing_pkg)

        companion_pkg = list()
        for pkg in faker.words(len(pkg_id_list)):
            companion_pkg.append({
                'package_name': pkg,
                'cooccurrence_probability': random.randint(30, 99),
                'topic_list': faker.words(random.randint(0, 3))
            })
        return companion_pkg, list(missing_pkg)

    def _map_input_to_package_ids(self, input_stack):
        """Filter out identified and missing packages."""
        kwown_packages = {'django', 'flask', 'werkzeug', 'six'}
        id_pkg = input_stack.intersection(kwown_packages)
        return id_pkg, input_stack.difference(id_pkg)


class MockAmazonS3(mock.Mock):
    """Mock AWS S3 class."""

    def connect(self):
        """Mock connect method."""
        pass


@pytest.fixture(scope='module')
@mock.patch('src.scoring.hpf_scoring.HPFScoring', new_callable=MockHPFScoring)
@mock.patch('rudra.data_store.aws.AmazonS3', new_callable=MockAmazonS3)
def api_client(_request, _awss3):
    """Create an api client instance."""
    from src.flask_endpoint import app
    client = app.app.test_client()
    return client


class TestFlaskMethods:
    """Test the flask endpoints with local data."""

    def test_root_path(self, api_client):
        """Test the root path."""
        resp = api_client.get('/')
        assert resp is not None
        assert resp.status_code == 200

    def test_liveness(self, api_client):
        """Test liveness endpoint."""
        resp = api_client.get('/api/v1/liveness')
        assert resp is not None
        assert resp.status_code == 200
        assert json.loads(resp.data.decode('UTF-8')) == {"status": "alive"}

    def test_readiness(self, api_client):
        """Test Readiness endpoint."""
        resp = api_client.get('/api/v1/readiness')
        assert resp is not None
        assert resp.status_code == 200
        assert json.loads(resp.data.decode('UTF-8')) == {"status": "ready"}

    def test_model_details(self, api_client):
        """Test model details endpoint."""
        resp = api_client.get('/api/v1/model_details')
        assert resp is not None
        assert resp.status_code == 200

    def test_companion_recommendations(self, api_client):
        """Test companion recommendations endpoint."""
        data = [{"ecosystem": "maven",
                 "package_list": ["org.springframework.cloud:spring-cloud-spring-service-connector"]
                 }]

        resp = api_client.post('/api/v1/companion_recommendation',
                               data=json.dumps(data),
                               headers={'content-type': 'application/json'})
        assert resp is not None
        assert resp.status_code == 200
