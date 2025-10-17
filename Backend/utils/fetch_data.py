import requests
import pandas as pd
import wikipedia
import os

def fetch_from_github(urls):
    os.makedirs("data", exist_ok=True)
    combined_text = ""

    for url in urls:
        print(f"📡 Fetching: {url}")
        try:
            r = requests.get(url)
            if r.status_code == 200:
                filename = url.split("/")[-1]
                path = os.path.join("data", filename)
                with open(path, "w", encoding="utf-8") as f:
                    f.write(r.text)

                if filename.endswith(".csv"):
                    df = pd.read_csv(path)
                    text_data = ""
                    for _, row in df.iterrows():
                        row_text = " | ".join([f"{col}: {str(row[col])}" for col in df.columns])
                        text_data += row_text + "\n"
                    combined_text += text_data
                else:
                    combined_text += r.text + "\n"
            else:
                print(f"❌ Failed: {url} (status {r.status_code})")
        except Exception as e:
            print(f"⚠️ Error: {e}")
    return combined_text

def fetch_from_wikipedia(topics):
    text_data = ""
    for topic in topics:
        try:
            summary = wikipedia.summary(topic, sentences=5)
            text_data += f"### {topic}\n{summary}\n\n"
        except:
            pass
    return text_data

def fetch_medical_data():
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

    with open("data/combined_medical_data.txt", "w", encoding="utf-8") as f:
        f.write(combined)
    print(f"✅ Combined data length: {len(combined)}")
    return combined
