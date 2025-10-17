# backend/utils/fetch_data.py
import os
import requests
import pandas as pd
import wikipedia

def fetch_from_github(urls):
    os.makedirs("data", exist_ok=True)
    combined_text = ""
    for url in urls:
        print(f"Fetching: {url}")
        try:
            r = requests.get(url, timeout=15)
            if r.status_code == 200:
                filename = url.split("/")[-1]
                path = os.path.join("data", filename)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(r.text)
                # If CSV, convert to friendly text
                if filename.endswith(".csv"):
                    try:
                        df = pd.read_csv(path)
                        rows_text = ""
                        for _, row in df.iterrows():
                            row_text = " | ".join([f"{col}: {str(row[col])}" for col in df.columns])
                            rows_text += row_text + "\n"
                        combined_text += rows_text + "\n\n"
                    except Exception as e:
                        print("CSV parse error:", e)
                        combined_text += r.text + "\n\n"
                else:
                    combined_text += r.text + "\n\n"
            else:
                print("Failed to fetch:", url, "status:", r.status_code)
        except Exception as e:
            print("Error fetching:", url, e)
    return combined_text

def fetch_from_wikipedia(topics, sentences=4):
    text = ""
    for t in topics:
        try:
            s = wikipedia.summary(t, sentences=sentences)
            text += f"### {t}\n{s}\n\n"
        except Exception as e:
            print("Wiki fetch failed for", t, e)
            continue
    return text

def fetch_medical_data():
    # Add or remove URLs as you wish
    github_urls = [
        "https://raw.githubusercontent.com/prasertcbs/disease-symptom-description-dataset/master/dataset.csv",
        "https://raw.githubusercontent.com/awesomedata/awesome-public-datasets/master/Health/README.md",
        "https://raw.githubusercontent.com/openmedlab/datahub/main/README.md"
    ]

    wiki_topics = [
        "Diabetes mellitus",
        "Hypertension",
        "Asthma",
        "Coronary artery disease",
        "Stroke",
        "Tuberculosis",
        "Anemia",
        "Malaria",
        "Vaccine",
        "Antibiotic"
    ]

    text_github = fetch_from_github(github_urls)
    text_wiki = fetch_from_wikipedia(wiki_topics)
    combined = text_github + "\n\n" + text_wiki

    os.makedirs("data", exist_ok=True)
    out_path = os.path.join("data", "combined_medical_data.txt")
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(combined)

    print("Combined data saved to", out_path)
    return combined
