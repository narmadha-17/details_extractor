
import os
import json
import time
import asyncio
import numpy as np
from collections import defaultdict
from typing import Optional
from pydantic import BaseModel, Field
from bs4 import BeautifulSoup
from PIL import Image, ImageChops
from selenium import webdriver
from selenium.webdriver.chrome.options import Options
from langchain.chat_models import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from tavily import AsyncTavilyClient

TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_api_key_here")
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your_api_key_here")

tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)
llm = ChatOpenAI(model="gpt-4.1", openai_api_key=OPENAI_API_KEY)

class InfoSearch(BaseModel):
    company_name: Optional[str] = Field(default=None)
    address_contact_information: Optional[str] = Field(default=None)
    company_size: Optional[str] = Field(default=None)
    type_of_industry: Optional[str] = Field(default=None)
    products_or_services: Optional[str] = Field(default=None)
    target_market: Optional[str] = Field(default=None)


def get_linkedin_logo(linkedin_url):
    chrome_options = Options()
    chrome_options.add_argument("--headless")
    chrome_options.add_argument("--disable-gpu")
    chrome_options.add_argument("--no-sandbox")
    chrome_options.add_argument('--disable-dev-shm-usage')

    driver = webdriver.Chrome(options=chrome_options)
    try:
        driver.get(linkedin_url)
        time.sleep(5)
        soup = BeautifulSoup(driver.page_source, 'html.parser')
        logo_img_tags = soup.find_all('img', {'data-delayed-url': lambda x: x and 'company-logo' in x})
        if logo_img_tags:
            return logo_img_tags[0]['data-delayed-url']
        logo_container = soup.find('div', class_='org-top-card-primary-content__logo-container')
        if logo_container:
            img_tag = logo_container.find('img')
            if img_tag and 'src' in img_tag.attrs:
                return img_tag['src']
    finally:
        driver.quit()
    return None

def compare_images_pil(image1_path, image2_path):
    try:
        img1 = Image.open(image1_path).convert("RGB")
        img2 = Image.open(image2_path).convert("RGB")
    except FileNotFoundError:
        return "Error: Image file not found."
    new_size = (max(img1.width, img2.width), max(img1.height, img2.height))
    img1 = img1.resize(new_size)
    img2 = img2.resize(new_size)
    diff = ImageChops.difference(img1, img2)
    diff_np = np.array(diff)
    total_diff = np.sum(diff_np)
    max_total_diff = new_size[0] * new_size[1] * 3 * 255
    match_percentage = ((max_total_diff - total_diff) / max_total_diff) * 100
    return match_percentage

def extract_info_from_url(url: str):
    try:
        print(f"Scraping URL: {url}")
        response = asyncio.run(tavily_client.extract(urls=[url], extract_depth="advanced"))
        content_blocks = [res.get("raw_content", "") for res in response.get("results", []) if res.get("raw_content")]
        joined_content = "\n".join(content_blocks).strip()
        if not joined_content:
            print("No content extracted.")
            return InfoSearch()

        # Try to infer company name from the URL for more accurate extraction
        import re
        domain_match = re.search(r"https?://(?:www\.)?([^/]+)", url)
        company_hint = None
        if domain_match:
            domain = domain_match.group(1)
            # Remove TLD and common subdomains
            company_hint = re.sub(r"(linkedin|zoominfo|crunchbase|youtube|www)\\.|\\..*", "", domain, flags=re.IGNORECASE)

        system_msg = SystemMessage(content=f"""
You are an assistant that extracts structured information about a company.
Given raw text about a company, return a JSON object with the following fields:
- company_name
- address_contact_information
- company_size
- type_of_industry
- products_or_services
- target_market

IMPORTANT: Only extract information about the company that matches the following hint (from the URL): '{company_hint if company_hint else url}'.
If there is information about other companies, ignore it. If you cannot find information about the company matching the hint, return an empty JSON object with all fields as null.
Only return a valid JSON object as output.
""".strip())

        human_msg = HumanMessage(content=f"""
Extract and return a JSON object with company information ONLY for the company matching this hint: '{company_hint if company_hint else url}'.
Ignore information about other companies. Use the following text:
{joined_content}
""".strip())

        llm_response = llm.invoke([system_msg, human_msg])
        print("LLM Response:", llm_response.content)

        json_text = llm_response.content.strip()
        if json_text.startswith("```json"):
            json_text = json_text[len("```json"):].strip()
        if json_text.endswith("```"):
            json_text = json_text[:-len("```")].strip()

        try:
            parsed_data = json.loads(json_text)
        except Exception as e:
            print(f"JSON parsing error: {e}")
            return InfoSearch()

        print("\n\n\nParsed Data:\n\n\n", parsed_data)
        return parsed_data
    except Exception as e:
        print(f"Error: {e}")
        return InfoSearch()

async def extract_content_from_url(url: str) -> str:
    try:
        print(f"Scraping URL: {url}")
        result = await tavily_client.extract(urls=[url], extract_depth="advanced")
        content = [res.get("raw_content", "") for res in result.get("results", []) if res.get("raw_content")]
        return "\n".join(content).strip()
    except Exception as e:
        print(f"Error scraping URL: {e}")
        return ""

async def process_company_search(company_name: str):
    try:
        search_result = await tavily_client.search(
            query=company_name,
            max_results=5,
            include_domains=["linkedin.com", "zoominfo.com", "youtube.com", "crunchbase.com"],
            include_answer=True
        )
        snippets = [res.get("content", "") for res in search_result.get("results", []) if res.get("content")]
        combined_text = "\n".join(snippets).strip()
        if not combined_text:
            return {"company_name": company_name, "info": "No content found"}

        system_msg = SystemMessage(content="""
You are an assistant that extracts structured information about a company.
Given raw text about a company, return a JSON object with the following fields:
- company_name
- address_contact_information
- company_size
- type_of_industry
- products_or_services
- target_market

Only return a valid JSON object as output.
""".strip())
        human_msg = HumanMessage(content=f"""
Extract and return a JSON object with company information based on the following text:
{combined_text}
""".strip())
        response = await llm.ainvoke([system_msg, human_msg])
        extracted = json.loads(response.content.strip())
        return {"company_name": company_name, "info": extracted}
    except Exception as e:
        return {"company_name": company_name, "info": f"Error: {e}"}

async def fetch_company_info_concurrently(company_name: str, company_url: list):
    if company_url:
        tasks = [extract_content_from_url(url) for url in company_url]
        scraped_contents = await asyncio.gather(*tasks, return_exceptions=True)
        extracted_info_list = []
        for i, content in enumerate(scraped_contents):
            if isinstance(content, Exception):
                extracted_info_list.append({"company_name": company_name, "info": f"Task failed for URL {company_url[i]} with error: {content}"})
            else:
                system_msg = SystemMessage(content="""
You are an assistant that extracts structured information about a company.
Given raw text about a company, return a JSON object with the following fields:
- company_name
- address_contact_information
- company_size
- type_of_industry
- products_or_services
- target_market

Only return a valid JSON object as output.
""".strip())
                human_msg = HumanMessage(content=f"""
Extract and return a JSON object with company information based on the following text:
{content}
""".strip())
                try:
                    response = await llm.ainvoke([system_msg, human_msg])
                    json_text = response.content.strip().strip('```json').strip('```')
                    extracted = json.loads(json_text)
                    extracted_info_list.append({"company_name": company_name, "info": extracted, "url": company_url[i]})
                except Exception as e:
                    extracted_info_list.append({"company_name": company_name, "info": f"Error processing content for URL {company_url[i]}: {e}", "url": company_url[i]})

        combined_info = defaultdict(dict)
        for item in extracted_info_list:
            company = item.get("company_name", "Unknown")
            info = item.get("info", {})
            url = item.get("url")
            if isinstance(info, dict):
                for key, value in info.items():
                    if value and not combined_info[company].get(key):
                        combined_info[company][key] = value
                if url:
                    combined_info[company].setdefault("urls", []).append(url)
            else:
                combined_info[company].setdefault("errors", []).append({"url": url, "message": info})
        return [{"company_name": company, "info": dict(info)} for company, info in combined_info.items()]
    else:
        return await asyncio.gather(process_company_search(company_name))
