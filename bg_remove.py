from rembg import remove
import easygui
from PIL import Image
import uuid

def generate_random_name():
    """Generate a random name using UUID."""
    return f"{uuid.uuid4()}.png"

# Open file dialog to select an image file
inputPath = easygui.fileopenbox(title='Select image file')

# Open the selected image
input_ = Image.open(inputPath)

# Remove the background
output_ = remove(input_)

# Generate a random name for the output file
output_name = generate_random_name()
outputPath = easygui.filesavebox(title='Save file to..', default=output_name)

if outputPath:
    # Save the output image in PNG format
    output_.save(outputPath, format='PNG')
