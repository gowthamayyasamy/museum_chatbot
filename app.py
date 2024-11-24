import gradio as gr
import google.generativeai as genai

# Option 1: Set the API key through environment variable
# Make sure to set GOOGLE_API_KEY as an environment variable on your system
# Or Option 2: Explicitly set the API key in the code
genai.configure(api_key="AIzaSyA9L8nQC2fwFUl_G-EyGMB1yhASbRIXP2Q")  # Replace with your actual API key

# Initialize the generative model
model = genai.GenerativeModel("gemini-1.5-flash")

# Define the function that will process the prompt and generate the response
def generate_response(prompt):
    # Generate response from the model
    response = model.generate_content([prompt])
    
    # Return the response text
    return response.text

# Create the Gradio interface (only with a Textbox input)
iface = gr.Interface(
    fn=generate_response, 
    inputs=gr.Textbox(label="Enter your prompt"),
    outputs=gr.Textbox(label="Generated Response"),
    live=True,
)

# Launch the Gradio app with the API enabled
iface.launch(share=True)
