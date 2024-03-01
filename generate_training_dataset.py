"""
This script is designed to interact with a local API to generate question-answer pairs from resume data. It uses requests to make HTTP calls, logging for error and information tracking, and includes functionality for extracting data from PDF resumes and optimizing prompts for a language model. The generated data is intended for use in training machine learning models.

It supports generating optimized prompts based on resume data, handling requests to a language model API, and creating a JSONL file containing the training data derived from the resume information.
"""

import requests
import json
import logging
from pdf_data_extraction import extract_data
from uk_resume_data_extraction import extract_resume_data

# Constants for API endpoint and model selection
URL = "http://localhost:11434/api/generate"
MODEL = "mistral"

# Configure logging for easy debugging and tracking
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize a requests session for efficient network calls
session = requests.Session()


def ask_to_llm(prompt):
    """
    Sends a prompt to a language model API and returns the response.

    Parameters:
    - prompt (str): The prompt to send to the language model.

    Returns:
    - dict: The JSON response from the API if the request was successful.

    Raises:
    - Exception: If the API request fails, it raises an exception with the error message.
    """
    payload = {"model": MODEL, "prompt": prompt, "stream": False}
    response = session.post(URL, json=payload)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"Error in generating instruction-answer pair: {response.text}")


def optimized_ask_to_llm(category, data, instruction):
    """
    Constructs a detailed prompt for the language model based on category, data, and a specific instruction, then queries the API.

    Parameters:
    - category (str): The category of the data to be included in the prompt.
    - data (str): The actual data related to the category.
    - instruction (str): Instruction to guide the generation of the prompt.

    Returns:
    - dict: The response from the language model API.
    """
    prompt = f"{instruction}\n" "---\n" "Data:\n" f"{category}: {data}\n"
    return ask_to_llm(prompt)


def generate_optimized_prompts(resume_data):
    """
    Generates optimized prompts for each category of resume data and collects the responses.

    Parameters:
    - resume_data (dict): A dictionary containing categorized resume data.

    Returns:
    - list: A list of dictionaries, each containing a text entry with the prompt and response.
    """
    categories_instructions = {
        "education": "Write a question about the person's educational background.",
        "experience": "Write a question about the person's professional experience.",
        "skills_interests_and_extracurricular_activities": "Write a question about the person's skills, interests, and extracurricular activities.",
        "key_achievements": "Write a question about the person's key achievements.",
        "personal_statement": "Write a question about the person's personal statement.",
        "contact_information": "Write a question about the person's contact information, including email, phone number, portfolio, and LinkedIn profile.",
    }
    results = []
    for category, data in resume_data.items():
        instruction = categories_instructions.get(
            category, "Write a question related to the data provided."
        )
        if data:
            prompt_response = optimized_ask_to_llm(category, data, instruction)
            answer_response = optimized_ask_to_llm(
                category,
                data,
                "Generate a concise answer in the third person to the following question, without making up any information.",
            )
            results.append(
                {
                    "text": f"<s>[INST] {prompt_response['response']}[/INST]{answer_response['response']}</s>"
                }
            )
    return results


def generate_training_data_jsonl(pdf_path, output_file, num_samples=1000):
    """
    Extracts data from a PDF resume, generates optimized prompts and answers, and writes the results to a JSONL file.

    Parameters:
    - pdf_path (str): The path to the PDF file containing the resume.
    - output_file (str): The file path where the JSONL data will be written.
    - num_samples (int, optional): The number of samples to generate. Default is 1000.

    This function does not return any value but writes directly to the file system.
    """
    try:
        resume_text = extract_data(pdf_path)
        resume_data = extract_resume_data(resume_text)

        for _ in range(num_samples):
            for result in generate_optimized_prompts(resume_data):
                with open(output_file, "a") as file:
                    file.write(json.dumps(result) + "\n")

    except Exception as e:
        logging.error(f"Error processing PDF or generating data: {e}")


# Example usage to demonstrate how the script can be employed
generate_training_data_jsonl(
    "/Users/alexis/Programmation/who-am-I/raw_data/UK_Resume_Alexis_BALAYRE.pdf",
    "train_data/training_data.jsonl",
    200,
)
