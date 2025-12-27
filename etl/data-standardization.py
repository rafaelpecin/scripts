import re
from typing import Dict, Any

def normalize_pii(data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalizes the SSN and phone number by stripping formatting characters,
    and ensures the full name is in Title Case for database insertion.

    Args:
        data: A dictionary containing PII fields.
    Returns:
        The dictionary with normalized values.
    """

    # 1. Standardize SSN and Phone Number: Use Regex to remove all non-digit characters,
    # but specifically keep the masking 'X' characters.
    if 'ssn' in data and data['ssn']:
        # Keep digits and 'X' only
        data['ssn'] = "".join(c for c in data['ssn'] if c.isdigit() or c == 'X')

    if 'phoneNumber' in data and data['phoneNumber']:
        # Remove all non-digit characters (for a clean, digit-only string)
        data['phoneNumber'] = re.sub(r'\D', '', data['phoneNumber'])

    # 2. Standardize Full Name: Ensure Title Case formatting
    if 'fullName' in data and data['fullName']:
        data['fullName'] = data['fullName'].title()

    # intakeDate is assumed to be standardized (YYYY-MM-DD)
    return data

# --- Test Case ---
raw_data = {
    "fullName": "eleanor vance", # Testing lowercase input
    "ssn": "XXX-77-9876",
    "phoneNumber": "(555) 234-7890",
    "intakeDate": "2024-09-23"
}

print("--- Raw Data ---")
print(raw_data)

normalized_result = normalize_pii(raw_data)
print("\n--- Normalized Data ---")
print(normalized_result)

