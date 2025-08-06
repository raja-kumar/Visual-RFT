import re

def pred_class_to_idx(cat_list, class_to_idx):
    idx_list = []
    for cat in cat_list:
        cat = cat.replace("'s", "")
        cat = re.sub(r'[^a-zA-Z0-9-]', ' ', cat)
        cat = cat.strip().lower()

        if ("barberton" in cat):
            cat = cat.replace("barberton", "barbeton")
        
        if cat in class_to_idx:
            idx_list.append(class_to_idx[cat])
        else:
            idx_list.append(-1)
    
    return idx_list

def clean_topk(topk_list, class_to_idx):
    """
    Cleans the top-k predictions by removing invalid entries and ensuring they are unique.
    """
    cleaned_topk = set()

    for item in topk_list:
        item = item.replace("'s", "")
        item = re.sub(r'[^a-zA-Z0-9-]', ' ', item)
        item = item.strip().lower()

        if ("barberton" in item):
            item = item.replace("barberton", "barbeton")

        if item in class_to_idx:
            cleaned_topk.add(item)
    
    return list(cleaned_topk)

def clean_string(text):
    """
    Cleans the input text by removing unwanted characters and formatting.
    """
    text = text.replace("'s", "")
    text = re.sub(r'[^a-zA-Z0-9-]', ' ', text)
    text = text.strip().lower()
    
    return text