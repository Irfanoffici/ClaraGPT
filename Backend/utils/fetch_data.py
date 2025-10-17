import requests
import os

def fetch_medical_data():
    urls = [
    "https://raw.githubusercontent.com/prasertcbs/disease-symptom-description-dataset/master/dataset.csv",
    "https://raw.githubusercontent.com/awesomedata/awesome-public-datasets/master/Health/README.md",
    "https://raw.githubusercontent.com/openmedlab/datahub/main/README.md"
]
    os.makedirs("data", exist_ok=True)
    combined_text = ""

    for url in urls:
        try:
            r = requests.get(url)
            if r.status_code == 200:
                text = r.text.strip()
                combined_text += text + "\n\n"
                with open(f"data/{url.split('/')[-1]}", "w", encoding="utf-8") as f:
                    f.write(text)
            else:
                print(f"Failed to fetch {url}")
        except Exception as e:
            print(f"Error fetching {url}: {e}")

    return combined_text
