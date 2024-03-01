"""
This script is designed for extracting various pieces of information from text, specifically targeting resume data. It employs regular expressions to extract contact information like emails and phone numbers, and utilizes the SpaCy library for more complex extractions such as names, education, and work experiences. The goal is to structure unstructured resume text into a more usable format for further processing or analysis.
"""

import re
import spacy
from spacy.matcher import Matcher


def extract_email(text):
    """
    Extracts the first occurring email address from the provided text.

    Parameters:
    - text (str): The text from which to extract an email address.

    Returns:
    - str or None: The extracted email address if found, otherwise None.
    """
    pattern = r"[\w\.-]+@[\w\.-]+"
    match = re.search(pattern, text)
    if match:
        return match.group()
    return None


def extract_phone_number(text):
    """
    Extracts the first occurring phone number from the provided text.

    Parameters:
    - text (str): The text from which to extract a phone number.

    Returns:
    - str or None: The extracted phone number if found, otherwise None.
    """
    pattern = r"[\+\(]?[1-9][0-9 .\-\(\)]{8,}[0-9]"
    match = re.search(pattern, text)
    if match:
        return match.group()
    return None


def extract_name(text):
    """
    Extracts the first occurring name from the provided text, assuming it's in uppercase.

    Parameters:
    - text (str): The text from which to extract a name.

    Returns:
    - str or None: The extracted name if found, otherwise None.
    """
    pattern = r"^\s*([A-Z]+\s+[A-Z]+)"
    match = re.search(pattern, text, re.MULTILINE)
    if match:
        return match.group(1)
    return None


def extract_portfolio_linkedin(text):
    """
    Extracts the first occurring portfolio or LinkedIn URL from the provided text.

    Parameters:
    - text (str): The text from which to extract a URL.

    Returns:
    - str or None: The extracted URL if found, otherwise None.
    """
    pattern = r"(https?://[^\s]+)|(linkedin\.com/[^\s]+)"
    match = re.search(pattern, text)
    if match:
        return match.group()
    return None


def get_nlp_doc(text):
    """
    Processes the provided text with SpaCy to create a SpaCy Document object.

    Parameters:
    - text (str): The text to process.

    Returns:
    - Doc: A SpaCy Document object.
    """
    nlp = spacy.load("en_core_web_trf") # en_core_web_trf is a transformer-based model
    return nlp(text)


def extract_categories(text):
    """
    Extracts and categorizes different sections of the resume based on predefined keywords using SpaCy's Matcher.

    Parameters:
    - text (str): The resume text.

    Returns:
    - dict: A dictionary with keys as categories and values as the extracted text for each category.
    """
    nlp = spacy.load("en_core_web_trf") # en_core_web_trf is a transformer-based model

    # Initialize the matcher with the shared vocab
    matcher = Matcher(nlp.vocab)

    # Define patterns for each category
    patterns = {
        "Education": [[{"LOWER": "education"}]],
        "Experience": [[{"LOWER": "experience"}]],
        "SKILLS, INTERESTS AND EXTRACURRICULAR ACTIVITIES": [
            [{"LOWER": "skills"}],
            [{"LOWER": "interests"}],
            [{"LOWER": "extracurricular"}, {"LOWER": "activities"}],
        ],
        "Key Achievements": [[{"LOWER": "key"}, {"LOWER": "achievements"}]],
        "Personal Statement": [[{"LOWER": "personal"}, {"LOWER": "statement"}]],
    }

    # Add patterns to the matcher
    for category, pattern in patterns.items():
        matcher.add(category, pattern)

    doc = get_nlp_doc(text)

    matches = matcher(doc)
    categories = {category: "" for category in patterns.keys()}
    start_pos = {category: None for category in patterns.keys()}
    for match_id, start, end in matches:
        rule_id = nlp.vocab.strings[match_id]  # Get the category name
        start_pos[rule_id] = start

    for category, pos in start_pos.items():
        if pos is not None:
            end_pos = len(
                doc
            )  # Default to end of doc if no further categories are found
            for other_cat, other_pos in start_pos.items():
                if other_pos is not None and other_pos > pos:
                    end_pos = min(end_pos, other_pos)
            span = doc[pos:end_pos]  # Extract the text span for this category
            categories[category] = span.text.strip().replace(
                category.upper() + "\n\n", ""
            )

    return categories


def extract_experiences(doc):
    """
    Extracts detailed work experiences from the 'Experience' section of a resume.

    Parameters:
    - doc (Doc): A SpaCy Document object containing the 'Experience' section text.

    Returns:
    - list: A list of dictionaries, each representing an extracted work experience.
    """
    nlp = spacy.load("en_core_web_trf") # en_core_web_trf is a transformer-based model

    doc = nlp(doc["Experience"])

    experiences = []
    index = -1
    for sent in doc.sents:
        labels = [ent.label_ for ent in sent.ents]

        if "DATE" in labels and "ORG" in labels and "GPE" in labels:
            experiences.append(
                {
                    "date": "",
                    "place": "",
                    "org-name": "",
                    "org-description": "",
                    "role": "",
                    "description": "",
                }
            )

            index += 1

            date = [ent.text for ent in sent.ents if ent.label_ == "DATE"]
            experiences[index]["date"] = date[0]

            org = [
                ent.text
                for ent in sent.ents
                if ent.label_ == "ORG" and "experience" not in ent.text.lower()
            ]
            experiences[index]["org-name"] = org[0]

            place = [ent.text for ent in sent.ents if ent.label_ == "GPE"]
            experiences[index]["place"] = place

            sent_split = sent.text.split("\n")
            sent_split = [
                s
                for s in sent_split
                if s != "" and s != " " and "experience" not in s.lower()
            ]
            role = sent_split[0].split(",")[3]
            role = role.replace(date[0], "")
            experiences[index]["role"] = role

            org_description = sent_split[1]
            experiences[index]["org-description"] = org_description

        else:
            if index >= 0:
                experiences[index]["description"] += sent.text.replace("\n", "") + " "
            else:
                print("No date, org, or place found in the sentence")

    return experiences


def extract_education(doc):
    """
    Extracts detailed education history from the 'Education' section of a resume.

    Parameters:
    - doc (Doc): A SpaCy Document object containing the 'Education' section text.

    Returns:
    - list: A list of dictionaries, each representing an extracted education history.
    """
    nlp = spacy.load("en_core_web_trf")

    doc = nlp(doc["Education"])

    education = []
    index = -1
    for sent in doc.sents:
        labels = [ent.label_ for ent in sent.ents]

        if "DATE" in labels and "ORG" in labels and "GPE" in labels:
            education.append(
                {
                    "date": "",
                    "place": "",
                    "institution": "",
                    "formation-name": "",
                    "description": "",
                }
            )

            index += 1

            date = [ent.text for ent in sent.ents if ent.label_ == "DATE"]
            education[index]["date"] = date[0]

            org = [
                ent.text
                for ent in sent.ents
                if ent.label_ == "ORG" and "education" not in ent.text.lower()
            ]
            education[index]["institution"] = org[1]

            place = [ent.text for ent in sent.ents if ent.label_ == "GPE"]
            education[index]["place"] = place

            sent_split = sent.text.split("\n")
            sent_split = [
                s
                for s in sent_split
                if s != "" and s != " " and "education" not in s.lower()
            ]
            formation_name = sent_split[0].split(",")[0]
            education[index]["formation-name"] = formation_name

            description = ""
            for i, word in enumerate(sent.text.split()):
                if word == "Modules:":
                    description = " ".join(sent.text.split()[i + 1 :])
                    break
            education[index]["description"] = "Modules: " + description + " "

        else:
            if index >= 0:
                education[index]["description"] += sent.text.replace("\n", "") + " "
            else:
                print("No date, org, or place found in the sentence")

    return education


def extract_resume_data(text):
    """
    Integrates all extraction functions to structure a resume's unstructured text into categorized data.

    Parameters:
    - text (str): The full text of the resume.

    Returns:
    - dict: A dictionary with structured resume data, categorized into education, experience, etc.
    """
    email = extract_email(text)
    phone_number = extract_phone_number(text)
    name = extract_name(text)
    portfolio_linkedin = extract_portfolio_linkedin(text)
    categories = extract_categories(text)

    data = {
        "education": f"Name: {name}\nEducation: {extract_education(categories)}",
        "experience": f"Name: {name}\nExperiences: {extract_experiences(categories)}",
        "skills_interests_and_extracurricular_activities": f"Name: {name}\nSkills, Interests, and Extracurricular Activities: {categories['SKILLS, INTERESTS AND EXTRACURRICULAR ACTIVITIES']}",
        "key_achievements": f"Name: {name}\nKey Achievements: {categories['Key Achievements']}",
        "personal_statement": f"Name: {name}\nPersonal Statement: {categories['Personal Statement']}",
        "contact_information": f"Name: {name}\nEmail: {email}\nPhone Number: {phone_number}\nPortfolio/LinkedIn: {portfolio_linkedin}",
    }

    return data
