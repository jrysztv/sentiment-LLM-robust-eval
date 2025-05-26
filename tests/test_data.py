"""
Tests for data loading and management components.
"""

import pytest
import pandas as pd
from unittest.mock import patch, Mock

from deep_learning_final_assignment.core.data.loaders import (
    SentimentLabels,
    DataSample,
    SST5Loader,
)


class TestSentimentLabels:
    """Test the SentimentLabels enum."""

    def test_sentiment_labels_values(self):
        """Test that sentiment labels have correct encodings."""
        assert SentimentLabels.VERY_NEGATIVE.encoding == -3
        assert SentimentLabels.NEGATIVE.encoding == -2
        assert SentimentLabels.NEUTRAL.encoding == 0
        assert SentimentLabels.POSITIVE.encoding == 2
        assert SentimentLabels.VERY_POSITIVE.encoding == 3

    def test_sentiment_labels_strings(self):
        """Test that sentiment labels have correct string representations."""
        assert SentimentLabels.VERY_NEGATIVE.label == "Very Negative"
        assert SentimentLabels.NEGATIVE.label == "Negative"
        assert SentimentLabels.NEUTRAL.label == "Neutral"
        assert SentimentLabels.POSITIVE.label == "Positive"
        assert SentimentLabels.VERY_POSITIVE.label == "Very Positive"

    def test_from_label(self):
        """Test creating sentiment enum from label string."""
        sentiment = SentimentLabels.from_label("Very Negative")
        assert sentiment == SentimentLabels.VERY_NEGATIVE
        assert sentiment.encoding == -3

    def test_from_label_invalid(self):
        """Test error handling for invalid label."""
        with pytest.raises(ValueError, match="Unknown sentiment label"):
            SentimentLabels.from_label("Invalid Label")

    def test_from_encoding(self):
        """Test creating sentiment enum from encoding."""
        sentiment = SentimentLabels.from_encoding(2)
        assert sentiment == SentimentLabels.POSITIVE
        assert sentiment.label == "Positive"

    def test_from_encoding_invalid(self):
        """Test error handling for invalid encoding."""
        with pytest.raises(ValueError, match="Unknown sentiment encoding"):
            SentimentLabels.from_encoding(99)

    def test_get_all_labels(self):
        """Test getting all sentiment labels."""
        labels = SentimentLabels.get_all_labels()
        expected = ["Very Negative", "Negative", "Neutral", "Positive", "Very Positive"]
        assert labels == expected

    def test_get_label_to_encoding_map(self):
        """Test label to encoding mapping."""
        mapping = SentimentLabels.get_label_to_encoding_map()
        expected = {
            "Very Negative": -3,
            "Negative": -2,
            "Neutral": 0,
            "Positive": 2,
            "Very Positive": 3,
        }
        assert mapping == expected

    def test_get_encoding_to_label_map(self):
        """Test encoding to label mapping."""
        mapping = SentimentLabels.get_encoding_to_label_map()
        expected = {
            -3: "Very Negative",
            -2: "Negative",
            0: "Neutral",
            2: "Positive",
            3: "Very Positive",
        }
        assert mapping == expected


class TestDataSample:
    """Test the DataSample dataclass."""

    def test_data_sample_creation(self):
        """Test creating a DataSample."""
        sample = DataSample(
            text="Test text", label="Positive", encoding=2, metadata={"length": 9}
        )

        assert sample.text == "Test text"
        assert sample.label == "Positive"
        assert sample.encoding == 2
        assert sample.metadata == {"length": 9}

    def test_data_sample_default_metadata(self):
        """Test DataSample with default metadata."""
        sample = DataSample(text="Test text", label="Positive", encoding=2)

        assert sample.metadata == {}

    def test_from_sst5_row(self):
        """Test creating DataSample from SST-5 row."""
        row = pd.Series(
            {
                "text": "This movie is great!",
                "label": 3,  # SST-5 positive label
            }
        )

        sample = DataSample.from_sst5_row(row)

        assert sample.text == "This movie is great!"
        assert sample.label == "Positive"
        assert sample.encoding == 2
        assert sample.metadata["original_sst5_label"] == 3
        assert sample.metadata["text_length"] == 20
        assert sample.metadata["word_count"] == 4

    def test_from_sst5_row_all_labels(self):
        """Test SST-5 label mapping for all labels."""
        test_cases = [
            (0, "Very Negative", -3),
            (1, "Negative", -2),
            (2, "Neutral", 0),
            (3, "Positive", 2),
            (4, "Very Positive", 3),
        ]

        for sst5_label, expected_label, expected_encoding in test_cases:
            row = pd.Series({"text": "Test text", "label": sst5_label})
            sample = DataSample.from_sst5_row(row)

            assert sample.label == expected_label
            assert sample.encoding == expected_encoding


class TestSST5Loader:
    """Test the SST5Loader class."""

    def test_init(self):
        """Test SST5Loader initialization."""
        loader = SST5Loader()

        expected_splits = {
            "train": "train.jsonl",
            "validation": "dev.jsonl",
            "test": "test.jsonl",
        }

        assert loader.splits == expected_splits
        assert loader.data == {}

    @patch("pandas.read_json")
    def test_load_split_success(self, mock_read_json, mock_sst5_data):
        """Test successful split loading."""
        mock_read_json.return_value = mock_sst5_data

        loader = SST5Loader()
        samples = loader.load_split("train")

        assert len(samples) == 5
        assert all(isinstance(sample, DataSample) for sample in samples)
        assert samples[0].text == "This movie is terrible!"
        assert samples[0].label == "Very Negative"

        # Check that data is cached
        assert "train" in loader.data
        assert len(loader.data["train"]) == 5

    def test_load_split_invalid(self):
        """Test loading invalid split."""
        loader = SST5Loader()

        with pytest.raises(ValueError, match="Unknown split"):
            loader.load_split("invalid_split")

    @patch("pandas.read_json")
    def test_load_split_cached(self, mock_read_json, mock_sst5_data):
        """Test that cached data is returned."""
        mock_read_json.return_value = mock_sst5_data

        loader = SST5Loader()

        # First call
        samples1 = loader.load_split("train")

        # Second call should use cache
        samples2 = loader.load_split("train")

        assert samples1 is samples2
        assert mock_read_json.call_count == 1

    @patch("pandas.read_json")
    def test_load_split_error(self, mock_read_json):
        """Test error handling during split loading."""
        mock_read_json.side_effect = Exception("Network error")

        loader = SST5Loader()

        with pytest.raises(RuntimeError, match="Failed to load SST-5"):
            loader.load_split("train")

    @patch("pandas.read_json")
    def test_load_all_splits(self, mock_read_json, mock_sst5_data):
        """Test loading all splits."""
        mock_read_json.return_value = mock_sst5_data

        loader = SST5Loader()
        all_data = loader.load_all_splits()

        assert len(all_data) == 3
        assert "train" in all_data
        assert "validation" in all_data
        assert "test" in all_data
        assert mock_read_json.call_count == 3

    @patch("pandas.read_json")
    def test_get_sample_subset_unbalanced(self, mock_read_json, mock_sst5_data):
        """Test getting unbalanced sample subset."""
        mock_read_json.return_value = mock_sst5_data

        loader = SST5Loader()
        samples = loader.get_sample_subset(
            split="train", n_samples=3, balanced=False, random_seed=42
        )

        assert len(samples) == 3
        assert all(isinstance(sample, DataSample) for sample in samples)

    @patch("pandas.read_json")
    def test_get_sample_subset_balanced(self, mock_read_json, mock_sst5_data):
        """Test getting balanced sample subset."""
        mock_read_json.return_value = mock_sst5_data

        loader = SST5Loader()
        samples = loader.get_sample_subset(
            split="train", n_samples=5, balanced=True, random_seed=42
        )

        assert len(samples) == 5

        # Check that we have one sample from each label
        labels = [sample.label for sample in samples]
        unique_labels = set(labels)
        assert len(unique_labels) == 5

    @patch("pandas.read_json")
    def test_get_label_distribution(self, mock_read_json, mock_sst5_data):
        """Test getting label distribution."""
        mock_read_json.return_value = mock_sst5_data

        loader = SST5Loader()
        distribution = loader.get_label_distribution("train")

        expected = {
            "Very Negative": 1,
            "Negative": 1,
            "Neutral": 1,
            "Positive": 1,
            "Very Positive": 1,
        }

        assert distribution == expected

    @patch("pandas.read_json")
    def test_get_text_length_stats(self, mock_read_json, mock_sst5_data):
        """Test getting text length statistics."""
        mock_read_json.return_value = mock_sst5_data

        loader = SST5Loader()
        stats = loader.get_text_length_stats("train")

        assert "mean_char_length" in stats
        assert "min_char_length" in stats
        assert "max_char_length" in stats
        assert "mean_word_count" in stats
        assert "min_word_count" in stats
        assert "max_word_count" in stats
        assert "total_samples" in stats

        assert stats["total_samples"] == 5
        assert isinstance(stats["mean_char_length"], float)

    @patch("pandas.read_json")
    def test_get_diverse_examples(self, mock_read_json, mock_sst5_data):
        """Test getting diverse examples."""
        mock_read_json.return_value = mock_sst5_data

        loader = SST5Loader()
        diverse_examples = loader.get_diverse_examples("train", n_per_label=1)

        assert len(diverse_examples) == 5  # One for each label

        for label in SentimentLabels.get_all_labels():
            assert label in diverse_examples
            assert len(diverse_examples[label]) == 1
            assert isinstance(diverse_examples[label][0], DataSample)
