# ClaraGPT Backend (Replit)

## Setup

1. Go to [Replit](https://replit.com/) → create a new Python web project.
2. Copy all files (`main.py`, `requirements.txt`, `.replit`) into your project.
3. Go to **Secrets** → add `GOOGLE_API_KEY` with your Gemini key.
4. Click **Run**.
5. Replit will give you a public URL: `https://<project-name>.<username>.repl.co`

## Test

POST JSON to `/ask`:

```bash
curl -X POST https://<project-name>.<username>.repl.co/ask \
  -H "Content-Type: application/json" \
  -d '{"q":"What is hypertension?"}'
