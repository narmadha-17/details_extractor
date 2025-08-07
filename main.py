import asyncio
import os
import json
import requests
import tempfile

from tavily import AsyncTavilyClient
from utils import (
    get_linkedin_logo,
    compare_images_pil,
    extract_info_from_url,
    fetch_company_info_concurrently,
    process_company_search
)

# API key should be in environment variable for security
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY", "your_api_key_here")
tavily_client = AsyncTavilyClient(api_key=TAVILY_API_KEY)


async def main():
    company_name = 'company_name'  # Replace with the actual company name
    print(f"Searching for company: {company_name}")
    local_logo_path = 'img/google.png'  # Path to the local logo file

    if not os.path.exists(local_logo_path):
        print(f"Error: Local logo file not found at {local_logo_path}.")
        return
    print(f"Using local logo file: {local_logo_path}")
    # Input URLs
    input_urls = ['your_linkedin_url_here', 'your_company_website_here']  # Replace with actual URLs
    # High-level URL validation
    from urllib.parse import urlparse
    def is_valid_url(url):
        try:
            result = urlparse(url)
            return all([result.scheme in ["http", "https"], result.netloc])
        except Exception:
            return False

    def is_linkedin_url(url):
        try:
            return "linkedin.com" in urlparse(url).netloc
        except Exception:
            return False

    def is_company_url(url, company_name):
        try:
            # Normalize company name for matching
            name_parts = company_name.lower().replace("private limited",""").replace("pvt ltd",""").replace("inc",""").replace("llc",""").split()
            url_lower = url.lower()
            return any(part in url_lower for part in name_parts if part)
        except Exception:
            return False

    # Validate input URLs
    print('\n Input URLs: \n', input_urls)
    print('\n Company Name: \n', company_name)

    valid_urls = [u for u in input_urls if is_valid_url(u)]
    linkedin_urls = [u for u in valid_urls if is_linkedin_url(u)]
    company_urls = [u for u in valid_urls if not is_linkedin_url(u)]

    if company_urls or linkedin_urls:
        # Process directly with validated URLs
        all_urls = company_urls + linkedin_urls
        company_info = await fetch_company_info_concurrently(company_name, all_urls)
        print("\n\n--- Final Extracted Content (From Provided URLs) ---\n\n")
    else:
        # Search LinkedIn URLs from Tavily
        response = await tavily_client.search(
            query=company_name,
            # max_results=5,
            include_domains=['linkedin.com']
        )

        linkedin_urls = [result['url'] for result in response['results'] if is_valid_url(result.get('url','')) and is_linkedin_url(result.get('url',''))]
        print("LinkedIn URLs:", linkedin_urls)

        comparison_results = []

        if os.path.exists(local_logo_path):
            print("Comparing local logo with logos from LinkedIn URLs...")
            for url in linkedin_urls:
                print(f"Processing URL: {url}")
                try:
                    logo_image_url = await asyncio.to_thread(get_linkedin_logo, url)

                    if logo_image_url:
                        print(f"Found logo image URL: {logo_image_url}")
                        res = requests.get(logo_image_url)
                        res.raise_for_status()

                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            tmp.write(res.content)
                            remote_image_path = tmp.name

                        match_percentage = compare_images_pil(local_logo_path, remote_image_path)
                        os.remove(remote_image_path)

                        if isinstance(match_percentage, (int, float)):
                            comparison_results.append((url, match_percentage))
                        else:
                            print(f"Error comparing images for {url}: {match_percentage}")

                    else:
                        print(f"Logo image URL not found on page: {url}")

                except requests.exceptions.RequestException as e:
                    print(f"Request error fetching logo image from {url}: {e}")
                except Exception as e:
                    print(f"Error processing image from {url}: {e}")

            comparison_results.sort(key=lambda x: x[1], reverse=True)

            if comparison_results:
                print("\nComparison Results (sorted by match percentage):")
                for url, match in comparison_results:
                    print(f"{url}: {match:.2f}%")

                matched_url = comparison_results[0][0]
                print(f"\nPotential best match: {matched_url}")

                print(f"Processing URL: {matched_url}")
                result = extract_info_from_url(matched_url)
                print("\n--- Structured InfoSearch Output ---")
                print(result)

                company_urls = [matched_url]
                company_info = await fetch_company_info_concurrently(company_name, company_urls)
            else:
                print("\nNo successful comparisons were made. Falling back to company name search...")
                company_info = await process_company_search(company_name)
        else:
            print(f"Error: Local logo file not found at {local_logo_path}. Falling back to company name search...")
            company_info = await process_company_search(company_name)

    if company_info:
        combined_company_data = company_info[0]
        print(json.dumps(combined_company_data['info'], indent=2))
        if 'errors' in combined_company_data['info']:
            print(f"Errors: {combined_company_data['info']['errors']}")
    else:
        print("No information found for the company.")

if __name__ == "__main__":
    asyncio.run(main())
