import datetime
import requests
import re
import uuid
import os
from pypdf import PdfReader
import pandas as pd
from dateutil import parser



def check_directory_exists(directory_path, create_if_not_exists=False):
    """
    Check if a directory exists. Optionally, create the directory if it does not exist.

    :param directory_path: Path of the directory to check.
    :param create_if_not_exists: If True, creates the directory if it does not exist.
    :return: True if the directory exists or was created, False otherwise.
    """
    if not os.path.isdir(directory_path):
        if create_if_not_exists:
            try:
                os.write(1,f"Directory does not exist: {directory_path}. Creating it.".encode())
                os.makedirs(directory_path)
                return True
            except OSError as error:
                os.write(1,f"Error creating directory {directory_path}: {error}".encode())
                return False
        else:
            os.write(1,f"Directory does not exist: {directory_path}".encode())
            return False
    return True



def get_most_recent_filepath_and_date(base_filename, directory_path, file_extension):
    """
    Returns the path of the most recent file based on the base filename, directory path, and file extension, along with its last modification date.

    This function searches for files matching the base filename pattern with the specified extension in the given directory and identifies the most recent file based on modification time. It also returns the modification time in 'dd Month YYYY' format.
        
    Usage:
        file_path, last_update_date = get_most_recent_file_path_and_date("library_catalog", "docs/library_catalog/", "xlsx")
    """

    check_directory_exists(directory_path, create_if_not_exists=True)
    files_in_directory = os.listdir(directory_path)
    # Construct regex pattern from base filename and file extension
    regex_pattern = rf'{base_filename}_\d{{4}}-\d{{2}}-\d{{2}}T\d{{4}}Z\.{file_extension}$'
    matching_files = [file for file in files_in_directory if re.match(regex_pattern, file)]

    if not matching_files:
        os.write(1, b"There's no matching file in the directory.\n")
        return None, None

    # Sort files based on the date-time in the filename
    matching_files.sort(key=lambda x: parser.parse(x[len(base_filename)+1:len(base_filename)+16]), reverse=True)
    most_recent_file = matching_files[0]
    last_update_date = most_recent_file[len(base_filename)+1:len(base_filename)+16]
    print(f"found the following file(s) {matching_files}")

    return os.path.join(directory_path, most_recent_file), last_update_date



def compute_doc_id(pdf_path):
    '''
    Generates a unique ID from the content of the PDF file.

    The function extracts text from all pages of the PDF--ignoring metadata-- and 
    generates a unique ID using UUID v5, example:  3b845a10-cb3a-5014-96d8-360c8f1bf63f 
    If the document is empty, then it sets the UUID to "EMPTY_DOCUMENT". 
    
    Args:
        pdf_path (str): Path to the PDF file.
    
    Returns:
        str: UUID for the PDF content or "EMPTY_DOCUMENT" if the PDF is empty.
    '''

    reader = PdfReader(download_folder)
    num_pages = len(reader.pages)

    # Extract text from all pages and concatenate
    full_text = ""
    for page_num in range(num_pages):
        try:
            page_text = reader.pages[page_num].extract_text()
            if page_text:
                full_text += page_text
        except Exception as e:
            logging.warning(f"Failed to extract text from page {page_num} of {pdf_path}: {e}")

    if not full_text.strip():
        return "EMPTY_DOCUMENT"

    namespace = uuid.NAMESPACE_DNS
    doc_uuid = uuid.uuid5(namespace, full_text)

    return doc_uuid

