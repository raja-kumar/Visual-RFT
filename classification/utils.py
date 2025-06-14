import json

def get_cat_name_from_json(json_file_path):
    """
    Get category names from a JSON file.
    
    Args:
        json_file_path (str): Path to the JSON file containing category names.
        
    Returns:
        list: A list of category names.
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)
    
    categories = [data[key] for key in data]

    print(f"Number of categories: {len(categories)}")
    print(categories)  # Print the category names
    return categories