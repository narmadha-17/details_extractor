# Smart Company Info Extractor

This Python script uses LinkedIn scraping, image comparison, and the Tavily Search API to extract structured company details from a set of input URLs—or by searching when none are provided.  It intelligently compares a local company logo to logos found on LinkedIn to identify the most accurate URL match before extracting company information. This project is an intelligent Python-based solution for extracting **structured company information** from web pages, primarily using **LinkedIn** and other business platforms (ZoomInfo, Crunchbase, etc.). It combines **web scraping**, **LLM-based extraction**, and **image similarity comparison** to identify the most relevant company information.

---

##  Key Features

-  **Smart URL validation and categorization** (LinkedIn vs Company URLs)
-  **Logo-based matching** between local image and remote LinkedIn logos
-  **Structured info extraction** using `GPT-4.1` via `LangChain`
-  **Tavily Search & Extraction API** integration for web search and content parsing
-  **Concurrent content scraping** and processing with asyncio
-  Modular utilities (`utils.py`) for scraping, comparing, and parsing


