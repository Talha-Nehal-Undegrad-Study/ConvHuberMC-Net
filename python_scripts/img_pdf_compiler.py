import os
from PIL import Image
from reportlab.lib.pagesizes import letter
from reportlab.pdfgen import canvas

def collect_png_files(root_directory, exception = False):
    png_files = []
    for Q in range(20, 81, 10):
        Q_folder = f"Q {Q}.0%"
        for DB in [3, 5, 6, 9]:
            DB_folder = f"DB {DB}.0"
            
            # Special condition for DB 6.0
            if exception:
                if DB == 6:
                    session_folder = os.path.join(root_directory, Q_folder, DB_folder)
            else:
                session_folder = os.path.join(root_directory, Q_folder, DB_folder, "Plots", "HuberMC-Net", "Session 3")
            
            # if os.path.exists(session_folder):
            #     for file in os.listdir(session_folder):
            #         print("Before: ", file, '\n')
            #         if file.endswith('.png') and '[1 out of 1]' in file:
            #             print("After: ", file, '\n')
            #             png_files.append((os.path.join(session_folder, file), Q, DB))
            # else:
            #     print(f"Directory not found: {session_folder}")

            if os.path.exists(session_folder):
                print(f"Checking directory: {session_folder}")  # Debug print
                for file in os.listdir(session_folder):
                    print("File found:", file)  # Debug print
                    if file.endswith('.png') and '[1_out_of_1]' in file:
                        print("Matched file:", file)  # Debug print
                        png_files.append((os.path.join(session_folder, file), Q, DB))
            else:
                print(f"Directory not found: {session_folder}")
    
            
    return png_files

def pngs_to_pdf(png_files, output_pdf):
    c = canvas.Canvas(output_pdf, pagesize=letter)
    width, height = letter

    for file_path, Q, DB in png_files:
        # Add text before the image
        text = f'Sampling Rate = {Q}%, DB = {DB}'
        c.drawString(100, height - 40, text)
        
        # Open the image
        img = Image.open(file_path)
        img_width, img_height = img.size
        
        # Resize image to fit the page width, maintaining aspect ratio
        aspect = img_height / float(img_width)
        img_width = width - 100
        img_height = img_width * aspect
        
        # Draw the image
        c.drawImage(file_path, 50, height - img_height - 60, width=img_width, height=img_height)

        # Move to the next page
        c.showPage()

    c.save()

# Example Usage
# root_directory = r'C:/Users/Talha/OneDrive - Higher Education Commission/Documents/GitHub/ConvHuberMC/HuberMC_Data'
# output_pdf = r'output.pdf'  # path for the output PDF

# png_files = collect_png_files(root_directory)
# pngs_to_pdf(png_files, output_pdf)