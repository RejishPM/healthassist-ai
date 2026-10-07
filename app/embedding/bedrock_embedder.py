import json

import boto3

from .embedder import Embedder


class BedrockEmbedder(Embedder):
    def __init__(
        self,
        model_id: str = "amazon.titan-embed-text-v2:0",
        region_name: str = "us-east-1",
    ):
        self.model_id = model_id

        self.client = boto3.client(
            "bedrock-runtime",
            region_name=region_name,
        )

    def get_embedding(self, text: str) -> list[float]:
        body = json.dumps(
            {
                "inputText": text,
            }
        )

        response = self.client.invoke_model(
            modelId=self.model_id,
            body=body,
            contentType="application/json",
            accept="application/json",
        )

        response_body = json.loads(
            response["body"].read()
        )

        return response_body["embedding"]