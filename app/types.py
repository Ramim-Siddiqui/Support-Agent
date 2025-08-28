from typing import TypedDict, List, Literal, Annotated
from operator import add

Category = Literal["Billing", "Technical", "Security", "General"]

class Review(TypedDict, total=False):
    approved: bool
    feedback: str

class State(TypedDict, total=False):
    # Input
    subject: str
    description: str

    # Derived
    category: Category
    queries: Annotated[List[str], add]
    context: Annotated[List[str], add]

    # Draft/review
    draft: str
    drafts: Annotated[List[str], add]
    review: Review
    reviews: Annotated[List[str], add]
    attempts: int

    # Logging
    history: Annotated[List[str], add]
    errors: Annotated[List[str], add]
