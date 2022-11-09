import unittest

import convert_to_sarif


class TestConvertToSarif(unittest.TestCase):
    def test_produce_sarif_returns_correct_sarif_result(self):
        lintrunner_results = [
            {
                "path": "test.py",
                "line": 1,
                "char": 2,
                "code": "FLAKE8",
                "severity": "error",
                "description": "test description",
                "name": "test-code",
            },
            {
                "path": "test.py",
                "line": 1,
                "char": 2,
                "code": "FLAKE8",
                "severity": "error",
                "description": "test description",
                "name": "test-code-2",
            },
            {
                "path": "test2.py",
                "line": 3,
                "char": 4,
                "code": "FLAKE8",
                "severity": "advice",
                "description": "test description",
                "name": "test-code",
            },
        ]
        actual = convert_to_sarif.produce_sarif(lintrunner_results)
        expected = {
            "$schema": "https://json.schemastore.org/sarif-2.1.0.json",
            "version": "2.1.0",
            "runs": [
                {
                    "tool": {
                        "driver": {
                            "name": "lintrunner",
                            "rules": [
                                {
                                    "id": "FLAKE8/test-code",
                                    "name": "FLAKE8/test-code",
                                    "shortDescription": {
                                        "text": "FLAKE8/test-code: test description"
                                    },
                                    "fullDescription": {
                                        "text": "FLAKE8/test-code\ntest description"
                                    },
                                    "defaultConfiguration": {"level": "note"},
                                },
                                {
                                    "id": "FLAKE8/test-code-2",
                                    "name": "FLAKE8/test-code-2",
                                    "shortDescription": {
                                        "text": "FLAKE8/test-code-2: test description"
                                    },
                                    "fullDescription": {
                                        "text": "FLAKE8/test-code-2\ntest description"
                                    },
                                    "defaultConfiguration": {"level": "error"},
                                },
                            ],
                        }
                    },
                    "results": [
                        {
                            "ruleId": "FLAKE8/test-code",
                            "level": "error",
                            "message": {"text": "FLAKE8/test-code\ntest description"},
                            "locations": [
                                {
                                    "physicalLocation": {
                                        "artifactLocation": {"uri": "test.py"},
                                        "region": {"startLine": 1, "startColumn": 2},
                                    }
                                }
                            ],
                        },
                        {
                            "ruleId": "FLAKE8/test-code-2",
                            "level": "error",
                            "message": {"text": "FLAKE8/test-code-2\ntest description"},
                            "locations": [
                                {
                                    "physicalLocation": {
                                        "artifactLocation": {"uri": "test.py"},
                                        "region": {"startLine": 1, "startColumn": 2},
                                    }
                                }
                            ],
                        },
                        {
                            "ruleId": "FLAKE8/test-code",
                            "level": "note",
                            "message": {"text": "FLAKE8/test-code\ntest description"},
                            "locations": [
                                {
                                    "physicalLocation": {
                                        "artifactLocation": {"uri": "test2.py"},
                                        "region": {"startLine": 3, "startColumn": 4},
                                    }
                                }
                            ],
                        },
                    ],
                }
            ],
        }
        self.maxDiff = None
        self.assertEqual(actual, expected)

    def test_it_handles_relative_paths(self):
        lintrunner_results = [
            {
                "path": "test.py",
                "line": 1,
                "char": 2,
                "code": "FLAKE8",
                "severity": "error",
                "description": "test description",
                "name": "test-code",
            },
        ]
        actual = convert_to_sarif.produce_sarif(lintrunner_results)
        expected_results = [
            {
                "ruleId": "FLAKE8/test-code",
                "level": "error",
                "message": {"text": "FLAKE8/test-code\ntest description"},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": "test.py"},
                            "region": {"startLine": 1, "startColumn": 2},
                        }
                    }
                ],
            },
        ]
        self.assertEqual(actual["runs"][0]["results"], expected_results)

    def test_it_handles_absolute_paths(self):
        lintrunner_results = [
            {
                "path": "/path/to/test.py",
                "line": 1,
                "char": 2,
                "code": "FLAKE8",
                "severity": "error",
                "description": "test description",
                "name": "test-code",
            },
        ]
        actual = convert_to_sarif.produce_sarif(lintrunner_results)
        expected_results = [
            {
                "ruleId": "FLAKE8/test-code",
                "level": "error",
                "message": {"text": "FLAKE8/test-code\ntest description"},
                "locations": [
                    {
                        "physicalLocation": {
                            "artifactLocation": {"uri": "file:///path/to/test.py"},
                            "region": {"startLine": 1, "startColumn": 2},
                        }
                    }
                ],
            },
        ]
        self.assertEqual(actual["runs"][0]["results"], expected_results)


if __name__ == "__main__":
    unittest.main()
