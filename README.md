# mmrag_qa

## Project Description
This is project aims to construct benchmark dataset MMRAG QA featuring in image-text interleaved response.

## Scripts Introduction
* **QA_construction/tool.py**: This script contains basic operations related to get responses from ChatGPT through API key.
* **QA_construction/wit_ConstructionTool.py**: This script specifies the pipeline of constructing QA with single image and corresponding context from WIT dataset.
* **QA_construction/web_ConstructionTool.py**: This script specifies the pipeline of constructing QA with multiple-images or single-image and corresponding contexts from WebQnA dataset.