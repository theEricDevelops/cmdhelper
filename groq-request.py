import os
import sys
import logging
from dotenv import load_dotenv

from groq import Groq

# Setup logger
logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)
formatter = logging.Formatter('%(levelname)s: %(name)s - %(message)s')

# Create file handler
file_handler = logging.FileHandler('groq.log')
file_handler.setLevel(logging.INFO)

# Set formatters
file_handler.setFormatter(formatter)

# Add handlers to logger
logger.addHandler(file_handler)

load_dotenv()

def get_command(request):

    client = Groq(api_key=os.getenv('GROQ_KEY'))

    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": """You are an expert linux developer. 
                            You will receive the request for a command and return only that command.
                            You will not return any additional information or context.
        
                            ---start
                            [USER]List all files in the current directory[/USER]
                            [RESPONSE]ls[/RESPONSE]
                            ---end

                            You will follow the format of that example exactly with the exception being you will not
                            include the [RESPONSE] tag in your response, only the command. 
                            Your response will not contain any formatting.
                            You will return only one command.
                            There is no option for more than one response so make it your best."""
            },
            {
                "role": "user",
                "content": request
            }
        ],
        model="mixtral-8x7b-32768",
    )
    logger.info(f"Response: {chat_completion}")

    return chat_completion.choices[0].message.content

if __name__ == "__main__":
    # Get the user's request from the command line argument
    request = ' '.join(sys.argv[1:])
    logger.info(f"Request: {request}")

    # Generate the command using the model
    command = get_command(request)

    # Print the command to stdout for the bash script to capture
    if command:
        print(command)
    else:
        print("Failed to generate command.")