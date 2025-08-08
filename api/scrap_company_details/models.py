from typing import Optional, List
from pydantic import BaseModel, Field


class ScrapCompanyDetailsRequest(BaseModel):
    domain_name: str = Field(description="Company name to refine the search")
    urls: Optional[List[str]] = Field(default=None, description="List of URLs to scrape for company details")
    logo: Optional[str] = Field(default=None, description="Path or ID of the logo to be used for the company details")


class SearchResponse(BaseModel):
    company_name: Optional[str] = Field(default=None, description="Name of the company")
    address_contact_information: Optional[str] = Field(default=None, description="Address and contact information")
    company_size: Optional[str] = Field(default=None, description="Size of the company")
    type_of_industry: Optional[str] = Field(default=None, description="Type of industry")
    products_or_services: Optional[str] = Field(default=None, description="Products or services offered")
    target_market: Optional[str] = Field(default=None, description="Target market")
    urls: Optional[List[str]] = Field(default=None, description="List of relevant URLs")