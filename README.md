Weviate Version


Run Streamlit App

Navigate to streamlit directory
Create `.streamlit/secrets.toml` file and set it with the proper values

```
OPENAI_API_KEY = ""
WCS_URL = ""
WCS_API_KEY = ""
COLLECTION_FULL_DOCUMENTS = "PDF_document"
COLLECTION_DOCUMENT_PAGES = "PDF_document_page"
```

Navigate into the `streamlit-app` directory

run the command:

```
streamlit run prompt_ui.py
```

