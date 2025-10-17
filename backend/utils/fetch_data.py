import requests
import pandas as pd
import wikipedia

# ✅ Cache results globally to avoid repeated fetching in serverless environment
_cached_medical_text = None

def fetch_from_github(urls):
    combined_text = ""

    for url in urls:
        print(f"📡 Fetching: {url}")
        try:
            r = requests.get(url, timeout=10)  # add timeout
            r.raise_for_status()
            filename = url.split("/")[-1]

            if filename.endswith(".csv"):
                df = pd.read_csv(pd.compat.StringIO(r.text))
                text_data = ""
                for _, row in df.iterrows():
                    row_text = " | ".join([f"{col}: {str(row[col])}" for col in df.columns])
                    text_data += row_text + "\n"
                combined_text += text_data
            else:
                combined_text += r.text + "\n"

        except Exception as e:
            print(f"⚠️ Error fetching {url}: {e}")
    return combined_text

def fetch_from_wikipedia(topics):
    text_data = ""
    for topic in topics:
        try:
            summary = wikipedia.summary(topic, sentences=5)
            text_data += f"### {topic}\n{summary}\n\n"
        except Exception as e:
            print(f"⚠️ Wiki error ({topic}): {e}")
    return text_data

def fetch_medical_data():
    global _cached_medical_text
    if _cached_medical_text is not None:
        return _cached_medical_text

    github_urls = [
        "https://raw.githubusercontent.com/prasertcbs/disease-symptom-description-dataset/master/dataset.csv",
        "https://raw.githubusercontent.com/awesomedata/awesome-public-datasets/master/Health/README.md"
    ]
    wiki_topics = [
        "Diabetes", "Cancer", "Asthma", "Hypertension", "COVID-19",
        "Heart Disease", "Stroke", "Tuberculosis", "Anemia", "Malaria",
        "Vaccine", "Antibiotics", "Immunity", "Nutrition", "Kidney Disease"
    ]

    github_text = fetch_from_github(github_urls)
    wiki_text = fetch_from_wikipedia(wiki_topics)
    combined = github_text + "\n\n" + wiki_text

    # Cache in memory
    _cached_medical_text = combined
    print(f"✅ Combined data length: {len(combined)}"
