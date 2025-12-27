import csv
import random
from datetime import datetime, timedelta
from faker import Faker

# 1. Configuration and Initialization
fake = Faker('en_US')
NUM_RECORDS = 10000
OUTPUT_FILE = 'mock/synthetic_customer_data.csv'

# Define the subscription plan distribution
PLANS = ['Free', 'Pro', 'Enterprise']
PLAN_WEIGHTS = [0.60, 0.30, 0.10] # 60% Free, 30% Pro, 10% Enterprise

# Define realistic usage score ranges based on the plan (Dependency D)
USAGE_RANGES = {
    'Free': (10, 45),  # Lower usage
    'Pro': (40, 80),   # Medium usage
    'Enterprise': (75, 100) # Highest usage
}

# 2. Main Data Generation Function
def generate_customer_record(customer_id):
    """Generates a single, realistic customer record based on dependencies."""
    
    # Randomly select a plan based on defined weights
    plan = random.choices(PLANS, weights=PLAN_WEIGHTS, k=1)[0]
    
    # Generate dates (Dependency D - Dates must be sequential)
    sign_up_date = fake.date_time_between(start_date='-2y', end_date='-3M')
    # Last login must be after sign-up, and recent (within last 3 months)
    last_login = fake.date_time_between(
        start_date=sign_up_date + timedelta(days=7), 
        end_date='now'
    )
    
    # Assign Usage Score based on plan (Dependency D)
    usage_min, usage_max = USAGE_RANGES[plan]
    usage_score = random.randint(usage_min, usage_max)

    # Calculate Account Age in Days
    account_age_days = (datetime.now() - sign_up_date).days
    
    # Calculate Monthly Recurring Revenue (MRR)
    if plan == 'Free':
        mrr = 0.00
    elif plan == 'Pro':
        mrr = round(random.uniform(49.99, 99.99), 2)
    else: # Enterprise
        mrr = round(random.uniform(500.00, 5000.00), 2)
        
    return {
        'CustomerID': f'AURA-{10000 + customer_id}',
        'Full_Name': fake.name(),
        'Email': fake.email(),
        'Subscription_Plan': plan,
        'Region': fake.country(),
        'Sign_Up_Date': sign_up_date.strftime('%Y-%m-%d'),
        'Last_Login': last_login.strftime('%Y-%m-%d %H:%M:%S'),
        'Usage_Score': usage_score, # Range 0-100
        'Account_Age_Days': account_age_days,
        'MRR': mrr,
        'Is_Active': random.choice([True, False, False]), # Weighted to be mostly active
    }

# 3. Execution and File Writing
if __name__ == '__main__':
    
    # Define the headers (Schema)
    fieldnames = [
        'CustomerID', 'Full_Name', 'Email', 'Subscription_Plan', 
        'Region', 'Sign_Up_Date', 'Last_Login', 'Usage_Score', 
        'Account_Age_Days', 'MRR', 'Is_Active'
    ]

    print(f"Generating {NUM_RECORDS} records for {OUTPUT_FILE}...")

    data_list = []
    for i in range(1, NUM_RECORDS + 1):
        data_list.append(generate_customer_record(i))
    
    try:
        with open(OUTPUT_FILE, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
            writer.writeheader()
            writer.writerows(data_list)
        print(f"Successfully created {OUTPUT_FILE}.")
    except Exception as e:
        print(f"An error occurred while writing the file: {e}")

