import asyncio
import logging
import os
import yaml
import httpx
from typing import List

# Configure logging
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(message)s")
logger = logging.getLogger(__name__)

# Example LLMClient Class
class LLMClient:
    def __init__(self, config_path: str):
        """Initialize LLMClient with configuration loaded from a YAML file."""
        # Step 1: Load configuration
        try:
            with open(config_path, 'r') as f:
                self.config = yaml.safe_load(f)
            logger.info(f"Configuration loaded successfully from {config_path}")
        except Exception as e:
            logger.error(f"Error loading configuration from {config_path}: {e}")
            raise

        # Step 2: Set up model and API URL
        self.model_name = self.config.get("model_name", "/kayla/openchat-3.5")
        self.base_url = self.config.get("openai_base_url", "http://localhost:8000/v1")
        self.api_key = os.getenv(self.config.get("llm", {}).get("api_key_env", "OPENAI_API_KEY"))

        if not self.api_key:
            logger.warning("API key not found in environment variables. Make sure it's set correctly.")
        
        # Log model name and base URL
        logger.info(f"Model Name: {self.model_name}, Base URL: {self.base_url}")

    async def query(self, prompt: str) -> str:
        """Query the LLM and return the response."""
        try:
            # Step 3: Make the API request
            logger.debug(f"Sending request to {self.base_url}/v1/chat/completions with prompt: {prompt}")
            async with httpx.AsyncClient(timeout=60.0) as client:
                response = await client.post(
                    f"{self.base_url}/v1/chat/completions",
                    json={
                        "model": self.model_name,
                        "messages": [{"role": "user", "content": prompt}],
                        "temperature": 0.7,
                        "max_tokens": 128,
                    },
                    headers={"Authorization": f"Bearer {self.api_key}"}
                )
                response.raise_for_status()  # Will raise an error if the status code is not 2xx
                response_data = response.json()
                logger.debug(f"Received response: {response_data}")
                return response_data["choices"][0]["message"]["content"]
        except Exception as e:
            logger.error(f"Error querying the LLM API: {e}")
            return f"Error: {e}"

async def debug_query_prompts_for_class(llm_client, prompts, class_name: str, dataset_name: str) -> tuple[str, List[str]]:
    """Debugging function to trace issues with querying the LLM."""
    all_concepts = []
    
    logger.debug(f"Starting query for class: {class_name}")

    # Step 1: Loop through prompts
    for idx, prompt_template in enumerate(prompts):
        try:
            prompt = prompt_template.format(class_name=class_name)
            logger.debug(f"Generated prompt {idx + 1}: {prompt}")
            
            # Step 2: Generate queries for each prompt
            tasks = [llm_client.query(prompt) for _ in range(10)]
            logger.debug(f"Generated {len(tasks)} tasks for queries.")
            
            # Step 3: Query LLM Client
            responses = await asyncio.gather(*tasks, return_exceptions=True)
            logger.debug(f"Received responses for prompt {idx + 1}")

            # Step 4: Process each response
            for response_idx, response in enumerate(responses):
                if isinstance(response, Exception):
                    logger.warning(f"⚠️ Error querying for {class_name} (Response {response_idx + 1}): {response}")
                    continue
                
                logger.debug(f"Parsing response {response_idx + 1} for {class_name}")
                try:
                    # Simulate response parsing (you can replace with actual parsing logic)
                    concepts = [f"Concept from {response}"]  # Replace with actual parsing logic
                    all_concepts.extend(concepts)
                    logger.debug(f"Parsed {len(concepts)} concepts from response {response_idx + 1}")
                except Exception as e:
                    logger.error(f"Error parsing response {response_idx + 1} for {class_name}: {e}")
        
        except Exception as e:
            logger.error(f"Error processing prompt {idx + 1} for class {class_name}: {e}")

    logger.debug(f"Completed query for class: {class_name}")
    return class_name, all_concepts

# Main function to test everything
async def main():
    # Step 5: Set the config path and initialize the LLM client
    config_path = 'config/query_config.yaml'  # Path to your actual config file
    if not os.path.exists(config_path):
        logger.error(f"Config file not found at {config_path}")
        return

    # Initialize LLM Client
    llm_client = LLMClient(config_path=config_path)

    # Test class and dataset names
    class_name = "frog"
    dataset_name = "cifar10"
    
    # Prompts to test
    prompts = [
        "Describe a {class_name} in detail.",
        "What are the characteristics of a {class_name}?",
        "List the features of a {class_name}."
    ]
    
    # Step 6: Run the debug function for queries
    await debug_query_prompts_for_class(llm_client, prompts, class_name, dataset_name)

# Run the async main function
if __name__ == "__main__":
    asyncio.run(main())
