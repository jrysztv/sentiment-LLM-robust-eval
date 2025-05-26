"""
Sentiment classification prompt variants.

Implements the 16 finalized prompt variants across 4 dimensions:
- Formality: Formal vs Casual
- Phrasing: Imperative vs Question
- Order: Task-first vs Text-first
- Synonyms: Set A (analyze/sentiment/classify) vs Set B (evaluate/emotion/categorize)
"""

from .template import PromptTemplate, PromptVariant


class SentimentPrompts(PromptTemplate):
    """Sentiment classification prompt template with 16 systematic variants."""

    def _setup_variants(self) -> None:
        """Setup all 16 prompt variants as specified in the research design."""

        # Formal + Imperative Variants (4)
        self.add_variant(
            PromptVariant(
                id="v1",
                name="Formal + Imperative + Task-first + Synonym A",
                template="""Analyze the sentiment of the following text and classify it as Very Negative, Negative, Neutral, Positive, or Very Positive. Return your response in JSON format with the key "sentiment" and the classified value.

Text: {input_text}

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "formal",
                    "phrasing": "imperative",
                    "order": "task_first",
                    "synonyms": "set_a",
                },
                description="Formal imperative with task description first, using 'analyze/sentiment/classify'",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v2",
                name="Formal + Imperative + Task-first + Synonym B",
                template="""Evaluate the emotion of the following text and categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive. Return your response in JSON format with the key "sentiment" and the classified value.

Text: {input_text}

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "formal",
                    "phrasing": "imperative",
                    "order": "task_first",
                    "synonyms": "set_b",
                },
                description="Formal imperative with task description first, using 'evaluate/emotion/categorize'",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v3",
                name="Formal + Imperative + Text-first + Synonym A",
                template="""Text: {input_text}

Analyze the sentiment of the above text and classify it as Very Negative, Negative, Neutral, Positive, or Very Positive. Return your response in JSON format with the key "sentiment" and the classified value.

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "formal",
                    "phrasing": "imperative",
                    "order": "text_first",
                    "synonyms": "set_a",
                },
                description="Formal imperative with text first, using 'analyze/sentiment/classify'",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v4",
                name="Formal + Imperative + Text-first + Synonym B",
                template="""Text: {input_text}

Evaluate the emotion of the above text and categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive. Return your response in JSON format with the key "sentiment" and the classified value.

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "formal",
                    "phrasing": "imperative",
                    "order": "text_first",
                    "synonyms": "set_b",
                },
                description="Formal imperative with text first, using 'evaluate/emotion/categorize'",
            )
        )

        # Formal + Question Variants (4)
        self.add_variant(
            PromptVariant(
                id="v5",
                name="Formal + Question + Task-first + Synonym A",
                template="""What is the sentiment of the following text? Please classify it as Very Negative, Negative, Neutral, Positive, or Very Positive and return your response in JSON format with the key "sentiment" and the classified value.

Text: {input_text}

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "formal",
                    "phrasing": "question",
                    "order": "task_first",
                    "synonyms": "set_a",
                },
                description="Formal question with task description first, using 'sentiment/classify'",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v6",
                name="Formal + Question + Task-first + Synonym B",
                template="""What is the emotion of the following text? Please categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive and return your response in JSON format with the key "sentiment" and the classified value.

Text: {input_text}

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "formal",
                    "phrasing": "question",
                    "order": "task_first",
                    "synonyms": "set_b",
                },
                description="Formal question with task description first, using 'emotion/categorize'",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v7",
                name="Formal + Question + Text-first + Synonym A",
                template="""Text: {input_text}

What is the sentiment of the above text? Please classify it as Very Negative, Negative, Neutral, Positive, or Very Positive and return your response in JSON format with the key "sentiment" and the classified value.

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "formal",
                    "phrasing": "question",
                    "order": "text_first",
                    "synonyms": "set_a",
                },
                description="Formal question with text first, using 'sentiment/classify'",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v8",
                name="Formal + Question + Text-first + Synonym B",
                template="""Text: {input_text}

What is the emotion of the above text? Please categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive and return your response in JSON format with the key "sentiment" and the classified value.

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "formal",
                    "phrasing": "question",
                    "order": "text_first",
                    "synonyms": "set_b",
                },
                description="Formal question with text first, using 'emotion/categorize'",
            )
        )

        # Casual + Imperative Variants (4)
        self.add_variant(
            PromptVariant(
                id="v9",
                name="Casual + Imperative + Task-first + Synonym A",
                template="""Check out this text and figure out the sentiment - is it Very Negative, Negative, Neutral, Positive, or Very Positive? Give me your answer in JSON format with "sentiment" as the key.

Text: {input_text}

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "casual",
                    "phrasing": "imperative",
                    "order": "task_first",
                    "synonyms": "set_a",
                },
                description="Casual imperative with task description first, using 'sentiment' terminology",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v10",
                name="Casual + Imperative + Task-first + Synonym B",
                template="""Look at this text and tell me the emotion - categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive. Give me your answer in JSON format with "sentiment" as the key.

Text: {input_text}

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "casual",
                    "phrasing": "imperative",
                    "order": "task_first",
                    "synonyms": "set_b",
                },
                description="Casual imperative with task description first, using 'emotion/categorize'",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v11",
                name="Casual + Imperative + Text-first + Synonym A",
                template="""Text: {input_text}

Check out this text above and figure out the sentiment - is it Very Negative, Negative, Neutral, Positive, or Very Positive? Give me your answer in JSON format with "sentiment" as the key.

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "casual",
                    "phrasing": "imperative",
                    "order": "text_first",
                    "synonyms": "set_a",
                },
                description="Casual imperative with text first, using 'sentiment' terminology",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v12",
                name="Casual + Imperative + Text-first + Synonym B",
                template="""Text: {input_text}

Look at this text above and tell me the emotion - categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive. Give me your answer in JSON format with "sentiment" as the key.

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "casual",
                    "phrasing": "imperative",
                    "order": "text_first",
                    "synonyms": "set_b",
                },
                description="Casual imperative with text first, using 'emotion/categorize'",
            )
        )

        # Casual + Question Variants (4)
        self.add_variant(
            PromptVariant(
                id="v13",
                name="Casual + Question + Task-first + Synonym A",
                template="""What's the sentiment of this text? Can you classify it as Very Negative, Negative, Neutral, Positive, or Very Positive and give me the answer in JSON format with "sentiment" as the key?

Text: {input_text}

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "casual",
                    "phrasing": "question",
                    "order": "task_first",
                    "synonyms": "set_a",
                },
                description="Casual question with task description first, using 'sentiment/classify'",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v14",
                name="Casual + Question + Task-first + Synonym B",
                template="""What's the emotion in this text? Can you categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive and give me the answer in JSON format with "sentiment" as the key?

Text: {input_text}

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "casual",
                    "phrasing": "question",
                    "order": "task_first",
                    "synonyms": "set_b",
                },
                description="Casual question with task description first, using 'emotion/categorize'",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v15",
                name="Casual + Question + Text-first + Synonym A",
                template="""Text: {input_text}

What's the sentiment of this text above? Can you classify it as Very Negative, Negative, Neutral, Positive, or Very Positive and give me the answer in JSON format with "sentiment" as the key?

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "casual",
                    "phrasing": "question",
                    "order": "text_first",
                    "synonyms": "set_a",
                },
                description="Casual question with text first, using 'sentiment/classify'",
            )
        )

        self.add_variant(
            PromptVariant(
                id="v16",
                name="Casual + Question + Text-first + Synonym B",
                template="""Text: {input_text}

What's the emotion in this text above? Can you categorize it as Very Negative, Negative, Neutral, Positive, or Very Positive and give me the answer in JSON format with "sentiment" as the key?

Response format: {{"sentiment": "Very Negative|Negative|Neutral|Positive|Very Positive"}}""",
                dimensions={
                    "formality": "casual",
                    "phrasing": "question",
                    "order": "text_first",
                    "synonyms": "set_b",
                },
                description="Casual question with text first, using 'emotion/categorize'",
            )
        )
