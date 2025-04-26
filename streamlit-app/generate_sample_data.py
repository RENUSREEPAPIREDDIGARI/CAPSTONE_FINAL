import pandas as pd

# Sample data
data = {
    'text': [
        "Discover our amazing collection of luxury watches",
        "Check out our new line of premium perfumes",
        "Experience luxury with our handcrafted bags",
        "Upgrade your style with our designer watches",
        "Find your signature scent with our exclusive perfumes",
        "Carry elegance with our premium handbags",
        "Timeless elegance in every watch",
        "Create lasting memories with our fragrances",
        "Style meets functionality in our bags",
        "Precision and luxury in every timepiece"
    ],
    'file_name': [
        "watch1.jpg", "perfume1.jpg", "bag1.jpg",
        "watch2.jpg", "perfume2.jpg", "bag2.jpg",
        "watch3.jpg", "perfume3.jpg", "bag3.jpg",
        "watch4.jpg"
    ],
    'local_path': [
        "/images/watch1.jpg", "/images/perfume1.jpg", "/images/bag1.jpg",
        "/images/watch2.jpg", "/images/perfume2.jpg", "/images/bag2.jpg",
        "/images/watch3.jpg", "/images/perfume3.jpg", "/images/bag3.jpg",
        "/images/watch4.jpg"
    ],
    's3_path': [
        "s3://images/watch1.jpg", "s3://images/perfume1.jpg", "s3://images/bag1.jpg",
        "s3://images/watch2.jpg", "s3://images/perfume2.jpg", "s3://images/bag2.jpg",
        "s3://images/watch3.jpg", "s3://images/perfume3.jpg", "s3://images/bag3.jpg",
        "s3://images/watch4.jpg"
    ],
    'platform': [
        "Instagram", "Twitter", "LinkedIn",
        "Instagram", "Twitter", "LinkedIn",
        "Instagram", "Twitter", "LinkedIn",
        "Instagram"
    ]
}

# Create DataFrame
df = pd.DataFrame(data)

# Save to CSV
df.to_csv('data_mapping.csv', index=False)
print("Sample data generated successfully!") 