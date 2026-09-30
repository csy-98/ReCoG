"""R2P MiniCPM components; see LICENSE for the upstream notice."""

import json

class BasePromptGenerator:
    def get_personalized_caption_prompt(self, test_image, concept_name, category_name, answer_format):
        prompt = f"""
        You are a helpful AI agent specializing in image analysis and caption generation.
        You are given an image and a concept name and a category name.
        Your task is to generate a personalized caption for the {category_name}, {concept_name} present in the image: {test_image}
        The caption must include the exact personalized concept name "{concept_name}" exactly as written.
        Do not replace "{concept_name}" with a species name, common object name, category name, or visual description.
        Write one concise sentence that naturally uses "{concept_name}" as the name of the visible {category_name}.
        Return only valid JSON in this exact format: {json.dumps(answer_format, indent=2)}
        """
        return prompt

    def get_image2image_plus_text_comparison_prompt(self, test_question, descriptions, answer_format):
        prompt = f"""
        You are a helpful AI agent specializing in image analysis and object recognition.
        You are given two images: **Image 1** and **Image 2**. Additionally, the name and a textual description of the object in **Image 2** is also provided below:
        {json.dumps(descriptions, indent=2)}
        **Task:**
        - Compare the two images and answer the following question.
        {test_question}
        """
        if "Reasoning" in answer_format:
            state_suffix = """
            - **Ignore superficial details** such as clothing, accessories, pose variations, or surrounding elements (e.g., people in the background). Focus only on non-variant/permanent features such as color, shape, pattern, text for objects/buildings and facial features for people.
            - If you are uncertain then you can refer the textual description of Image 2 to make a more informed decision.\n
            """
            reason_suffix = f"""
            - Provide your reasoning based on the two images and the given description.
            - Generate your response with JSON format {json.dumps(answer_format, indent=2)}. Output only the JSON response. DO NOT output any additional text.
            """
            suffix = state_suffix + reason_suffix
        else:
            suffix = f"""
             - Generate your response with JSON format {json.dumps(answer_format, indent=2)}. Output only the JSON response. DO NOT output any additional text.
            """
        prompt += suffix
        return prompt

    def get_text_options_prompt(self, test_question, descriptions, answer_format):
        prompt = f"""You are a helpful AI agent specializing in image analysis and object recognition.
        You are provided with a query image along with detailed description(s) of one or several objects.
        Below are the description(s):
        {json.dumps(descriptions, indent=2)}
        Your Task:
        - Compare the query image with the provided description(s) and answer the following question:
        {test_question}
        """
        if "Reasoning" in answer_format:
            state_suffix = """
            - **Ignore superficial details** such as clothing, accessories, pose variations, or surrounding elements (e.g., people in the background). Focus only on non-variant/permanent features such as color, shape, pattern, text for objects/buildings and facial features for people.
            """
            reason_suffix = f"""
            - Please provide a reasoning for your answer generate your response with JSON format {json.dumps(answer_format)}.
            Any deviation from this format will be considered incorrect. Output only the JSON response, without any additional text.
            """
            suffix = state_suffix + reason_suffix
        else:
            suffix = f"""
            - Generate your response with JSON format {json.dumps(answer_format, indent=2)}. Output only the JSON response. DO NOT output any additional text.
            """
        prompt += suffix
        return prompt

    def get_attribute_based_text_options_prompt(self, test_question, descriptions, answer_format):
        prompt = f"""You are a helpful AI agent specializing in image analysis and object recognition.
        Your task is to analyze a query image and compare it with three provided descriptions.
        Below are the description(s):
        {json.dumps(descriptions, indent=2)}
        Your Task:
        - Compare the query image with each description and answer the following question:
        {test_question}
        """
        if "Reasoning" in answer_format:
            state_suffix = """
            - **Ignore superficial details** such as clothing, accessories, pose variations, or surrounding elements (e.g., people in the background). Focus only on non-variant/permanent features such as color, shape, pattern, text for objects/buildings and facial features for people.
            """
            reason_suffix = f"""""
            - List shared attributes between the image and each description very concisely (at max 5 words).
            - Provide a brief reasoning for your final answer.
            - Respond strictly in the following JSON format:
            {json.dumps(answer_format, indent=2)}
            Any deviation from this format will be considered incorrect. Do not output any additional text.
            """
            suffix = state_suffix + reason_suffix
        else:
            suffix = f"""
            - Generate your response with JSON format {json.dumps(answer_format, indent=2)}. Output only the JSON response. DO NOT output any additional text.
            """
        prompt += suffix
        return prompt
