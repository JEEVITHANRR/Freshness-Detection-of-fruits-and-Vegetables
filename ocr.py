import cv2
import pytesseract

# Optional: specify the path to Tesseract executable if not in PATH
# pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

def extract_text(image_path):
    # Read the image
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")

    # Convert to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Optional: apply thresholding or noise removal for better OCR
    gray = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]

    # Extract text using pytesseract
    text = pytesseract.image_to_string(gray, lang='eng')  # specify language if needed

    return text

# Example usage:
if __name__ == "__main__":
    image_path = "test_image.jpg"
    extracted_text = extract_text(image_path)
    print("Extracted Text:\n", extracted_text)
