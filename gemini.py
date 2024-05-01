import sys
import logging
import base64
import vertexai
from vertexai.generative_models import GenerativeModel, Part, FinishReason
import vertexai.preview.generative_models as generative_models

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(name)s - %(message)s')

# Create file handler
file_handler = logging.FileHandler('gemini.log')
file_handler.setLevel(logging.INFO)

# Set formatters
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)

def get_command(request):
  vertexai.init(project="ericdevelops-development", location="us-east1")
  model = GenerativeModel("gemini-1.5-pro-preview-0409")

  logger.info(f"Received request: {request} for {model}")

  generation_config = {
    "max_output_tokens": 8192,
    "temperature": 0.5,
    "top_p": 0.95,
}

  safety_settings = {
      generative_models.HarmCategory.HARM_CATEGORY_HATE_SPEECH: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      generative_models.HarmCategory.HARM_CATEGORY_DANGEROUS_CONTENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      generative_models.HarmCategory.HARM_CATEGORY_SEXUALLY_EXPLICIT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
      generative_models.HarmCategory.HARM_CATEGORY_HARASSMENT: generative_models.HarmBlockThreshold.BLOCK_MEDIUM_AND_ABOVE,
  }

  request_message= """[SYS]You are an expert linux developer. 
    You will receive the request for a command and return only that command.
    You will not return any additional information or context.
    
    ---start
    [USER]List all files in the current directory[/USER]
    [RESPONSE]ls[/RESPONSE]
    ---end

    You will follow the format of that example exactly with the exception being you will not
    include the [RESPONSE] tag in your response, only the command. You will return only one command.
    There is no option for more than one response so make it your best.
    [/SYS][USER]{request}[/USER]""".format(request=request)
  
  logger.info(f"Request message: {request_message}")

  try:
    responses = model.generate_content(
        [request_message],
        generation_config=generation_config,
        safety_settings=safety_settings,
        stream=False,
    )

    logger.info(f"Response: {responses}")

    return responses[0].content.parts.text

  except Exception as e:
    print(f"Error calling Vertex AI API: {e}")
    return None

if __name__ == "__main__":
    # Get the user's request from the command line argument
    request = sys.argv[1]
    logger.info(f"Request: {request}")

    # Generate the command using the model
    command = get_command(request)

    # Print the command to stdout for the bash script to capture
    if command:
        print(command)
    else:
        print("Failed to generate command.")