import asyncio
import os
import re
import tempfile
import requests

from fastapi import APIRouter, Request, Depends, HTTPException
from fastapi.responses import JSONResponse

from tavily import AsyncTavilyClient

from api.scrap_company_details.models import ScrapCompanyDetailsRequest
from api.scrap_company_details.utils import (
    get_linkedin_logo,
    compare_images_pil,
    extract_info_from_url,
    fetch_company_info_concurrently,
    process_company_search,
)

from api.vector_db.utils import write_file_into_local
from common_utils.logging.fastapi.middlewares import custom_ml_response_logging
from common_utils.utils import get_signed_url
from common_utils.shared_utils.dependency_utils import set_developer_provider_config


router = APIRouter()


@router.post("/scrap_company_details")
@custom_ml_response_logging
async def scrap_company_details(
    request: Request,
    payload: ScrapCompanyDetailsRequest,
    external_tool_config: dict = Depends(set_developer_provider_config),
):
    web_search_config = external_tool_config.get("web_search_config", {})
    tavily_api_key = web_search_config.get("key", {}).get("api_key")
    if not tavily_api_key:
        raise HTTPException(status_code=500, detail="Missing Tavily API key in configuration")

    tavily_client = AsyncTavilyClient(api_key=tavily_api_key)

    company_name = payload.domain_name
    company_urls = payload.urls
    logo_id = payload.logo
    local_logo_path = None

    metadata = None
    try:
        body = await request.json()
        metadata = body.get("metadata")
    except Exception:
        metadata = None

    # Resolve logo path if provided
    if logo_id:
        if re.fullmatch(r"[a-fA-F0-9]{24}", str(logo_id)):
            if metadata is not None:
                document_details = {
                    "user_id": metadata.get("user_id"),
                    "username": metadata.get("username"),
                    "is_system_generated_doc": True,
                    "document_id": [{"id": logo_id}],
                }
                signed_url = get_signed_url(metadata=metadata, document_details=document_details)
                file_url = signed_url["document_url"]
                file_path = write_file_into_local(file_url)
                local_logo_path = file_path
            else:
                raise HTTPException(status_code=400, detail="Metadata required to resolve logo file.")
        else:
            # Assume direct local path
            local_logo_path = str(logo_id)

        if not local_logo_path or not os.path.exists(local_logo_path) or str(local_logo_path).startswith("http"):
            raise HTTPException(status_code=400, detail=f"Logo file not found at {local_logo_path}")

    # If URLs are provided, scrape directly
    if company_urls:
        company_info = await fetch_company_info_concurrently(company_name, company_urls, external_tool_config)
    else:
        # Search LinkedIn URLs via Tavily
        response = await tavily_client.search(query=company_name, include_domains=["linkedin.com"])  # type: ignore
        linkedin_urls = [result.get("url") for result in response.get("results", []) if result.get("url")]

        comparison_results = []

        if local_logo_path and os.path.exists(local_logo_path):
            for url in linkedin_urls:
                try:
                    logo_image_url = await asyncio.to_thread(get_linkedin_logo, url)
                    if logo_image_url:
                        res = requests.get(logo_image_url, timeout=15)
                        res.raise_for_status()
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".png") as tmp:
                            tmp.write(res.content)
                            remote_image_path = tmp.name
                        match_percentage = compare_images_pil(local_logo_path, remote_image_path)
                        os.remove(remote_image_path)
                        if isinstance(match_percentage, (int, float)):
                            comparison_results.append((url, float(match_percentage)))
                except requests.exceptions.RequestException:
                    continue
                except Exception:
                    continue

            comparison_results.sort(key=lambda x: x[1], reverse=True)

            if comparison_results:
                matched_url = comparison_results[0][0]
                # Extract and fetch details from best-matched LinkedIn URL
                _ = await extract_info_from_url(matched_url, external_tool_config)
                company_urls = [matched_url]
                company_info = await fetch_company_info_concurrently(company_name, company_urls, external_tool_config)
            else:
                company_info = await process_company_search(company_name, external_tool_config)
        else:
            company_info = await process_company_search(company_name, external_tool_config)

    if company_info:
        combined_company_data = company_info[0]
        response_data = combined_company_data.get("info", {}) if isinstance(combined_company_data, dict) else {}
        if isinstance(response_data, dict) and "errors" in response_data:
            # Log errors if needed
            pass
        extracted_result = JSONResponse(content=response_data)
    else:
        extracted_result = JSONResponse(content={"detail": "No information found for the company."}, status_code=404)

    return extracted_result