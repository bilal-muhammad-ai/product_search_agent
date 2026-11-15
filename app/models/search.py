from pydantic import BaseModel, Field


class RewrittenQuery(BaseModel):
    """Product attributes pulled from a customer search message."""

    search_query: str = Field(
        default="",
        description="The core product search query extracted from the customer's message.",
    )
    item: str = Field(
        default="",
        description="Product type the customer asked for, such as sofa or dining chair. Empty if not stated.",
    )
    brand: str = Field(
        default="",
        description="Brand name the customer stated. Empty if not stated. Never guess a brand.",
    )
    colour: str = Field(
        default="",
        description="Colour in the customer's own words. Empty if not stated. Do not invent a catalogue colour name.",
    )
    material: str = Field(
        default="",
        description="Material the customer stated, such as leather or oak. Empty if not stated. Never guess a material.",
    )
    range: str = Field(
        default="",
        description="Product range or collection the customer stated. Empty if not stated. Never guess a range.",
    )


class RetrievedProduct(BaseModel):
    id: str
    score: float
    pr_item: str | None = None
    pr_brand: str | None = None
    pr_colour: str | None = None
    pr_material: str | None = None
    pr_range: str | None = None


class SearchRequest(BaseModel):
    message: str = Field(min_length=1, max_length=4000)


class SearchResponse(BaseModel):
    reply: str
    item: str = ""
    brand: str = ""
    colour: str = ""
    material: str = ""
    range: str = ""
    products: list[RetrievedProduct] = []

class SearchProductsResponse1(BaseModel):
    count: int
    next_after_id: str | None
    products: list[RetrievedProduct]
