# Standard library imports
from pathlib import Path

# third party library imports
from fastapi import APIRouter, Request

# local imports
from src.loaders.text_loader import process_text, process_pdf, process_docx
from src.loaders.image_loader import process_image
from src.loaders.video_loader import process_video

# Create a router for the parser endpoints
router = APIRouter(tags=["data_loader"])

@router.post("/load-data")
def add_knowledge(request: Request, folder_path: str="data"):
    print(f'Request data received {request.json()}')
    for file in Path(folder_path).glob("*"):
        if file.suffix.lower() == ".txt":
            print(f"Processing text file: {file}")
            process_text(file)
        elif file.suffix.lower() in [".pdf"]:
            print(f"Processing PDF file: {file}")
            process_pdf(file)
        # elif file.suffix.lower() in [".jpg", ".jpeg", ".png"]:
        #     process_image(file)
        # elif file.suffix.lower() in [".docx"]:
        #     print(f"Processing DOCX file: {file}")
        #     process_docx(file)
        # elif file.suffix.lower() in [".mp4", ".mov", ".avi"]:
        #     process_video(file)
        else:
            print(f"Unsupported file type: {file}")

    return {"message": "Files processed and added to knowledge base!"}
