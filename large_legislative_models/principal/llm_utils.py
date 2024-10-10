import os
import re

import google.generativeai as genai
from dotenv import load_dotenv
from google.api_core.exceptions import ResourceExhausted
import openai
from abc import ABC, abstractmethod

def api_handler_factory(model_name, temperature):
    if "gemini" in model_name:
        return GeminiHandler(model_name, temperature)
    elif "gpt" in model_name:
        return GPTHandler(model_name, temperature)
    else:
        raise ValueError(f"Unknown handler type: {model_name}")
    
class APIHandler(ABC):
    def __init__(self, model_name, temperature):
        self.model_name = model_name
        self.temperature = temperature
    """ The following abstract methods are model-specific"""
    @abstractmethod
    def load_model(self, *args, **kwargs):
        pass

    @abstractmethod
    def get_response(self, *args, **kwargs):
        pass  

    @abstractmethod
    def query_llm(self, *args, **kwargs):
        pass
    
    def parse_and_validate_response(self, input_string, num_brackets):
        """
        This will parse the response from the model and validate that it.
        We prompt the LLM to output tax rates / incentives wrapped in dollar signs, and these
        are used here to for parsing the action out of response.
        """
        input_string = re.sub(r'\$\$', '$', input_string)

        # Define the regex pattern to find the part wrapped in dollar signs
        pattern = r"\$(.*?)\$"

        # Search for the pattern in the input string
        match = re.search(pattern, input_string)

        if match:
            # Extract the part within the dollar signs
            dollar_content = match.group(1).strip()
            dollar_content = dollar_content.replace("*", "")

            # Try to convert the content to a list of floats
            try:
                # Remove any surrounding brackets and split the content by space or comma
                dollar_content = re.sub(r"[\[\]\{\}\(\)]", "", dollar_content)
                float_list = list(map(float, filter(None, re.split(r"[,;\s]+", dollar_content))))
                assert len(float_list) == num_brackets
                print("Valid list of floats:", float_list)
                return float_list
            except ValueError as ve:
                error_msg = "Error: The content within the dollar signs is not a valid list of floats."
                print(error_msg)
                raise ValueError(error_msg) from ve
            except AssertionError:
                print(dollar_content)
                error_msg = f"Error: The list must contain exactly {num_brackets} floats."
                print(error_msg)
                raise ValueError(error_msg)
        else:
            error_msg = "Error: No content found within dollar signs."
            print(error_msg)
            raise ValueError(error_msg)


class GeminiHandler(APIHandler):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        self.model = self.load_model()

    def load_model(self):
        load_dotenv()
        api_key = os.getenv("GEMINI_API_KEY")
        genai.configure(api_key=api_key)
        model = genai.GenerativeModel(self.model_name)
        return model

    def get_response(self, prompt):
        try:
            response = self.model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(temperature=self.temperature),
            )
            return response
        except ResourceExhausted:
            print(f"API key exhausted")
            raise
    
    def query_llm(self, prompt):
        unvalidated = self.get_response(prompt)
        """The output from the API contains more than just the natural language response,
        so we need to extract the text from the response that the model generated."""
        unvalidated_response = unvalidated.candidates[0].content.parts[0].text
        return unvalidated_response


class GPTHandler(APIHandler):
    def __init__(self, model_name, temperature):
        super().__init__(model_name, temperature)
        self.client = self.load_model()
    def load_model(self):
        client = openai.OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
        )
        return client

    def get_response(self, prompt):
        try:
            response = self.client.chat.completions.create(
                messages=[
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ],
                model=self.model_name,
                temperature=self.temperature,
            )
        except Exception as e:
            print(f"Error: {e}")
            raise
        return response

    def query_llm(self, prompt):
        response = self.get_response(prompt)
        txt = response.choices[0].message.content
        return txt
    
def retry_with_prompt_adjustment(max_attempts=3, reminder="Make sure to format your output as requested. "):
    def decorator(func):
        def wrapper(*args, **kwargs):
            prompt = kwargs['prompt']  # expects "prompt" to be passed in kwargs
            for attempt in range(1, max_attempts + 1):
                try:
                    return func(*args, **kwargs)
                except Exception as e:
                    print(f"Attempt {attempt} failed: {e}")
                    last_exception = e
                    if attempt < max_attempts:
                        prompt += reminder
                        kwargs['prompt'] = prompt
            raise Exception("Failed to generate a valid LLM response after retries.") from last_exception
        return wrapper
    return decorator


def format_demonstration(env_name, gets_historical_obs, generation_num, action, reward, historical_obs):
    natural_language_historical_obs = ""
    match env_name:
        case "commons_harvest__open":
            if gets_historical_obs:
                natural_language_historical_obs = "SOME FUNCTION OF historical_obs"
            return f"""Generation {generation_num}: {action} -> reward: {int(reward)}. {natural_language_historical_obs} \n"""
        case "clean_up":
            if gets_historical_obs:
                natural_language_historical_obs = "SOME FUNCTION OF historical_obs"
            return f"""Generation {generation_num}: {action} -> reward: {int(reward)}. {natural_language_historical_obs} \n"""
        case "cer":
            if gets_historical_obs:
                natural_language_historical_obs = "SOME FUNCTION OF historical_obs"
            return f"""Generation {generation_num}: {action} -> reward: {int(reward)}. {natural_language_historical_obs} \n"""
