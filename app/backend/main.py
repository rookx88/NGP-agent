from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from .db import init_db

app = FastAPI()

# Allow frontend (Streamlit or Next.js) to access API
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def on_startup():
    init_db()

@app.get("/health")
def health_check():
    return {"status": "ok"}

@app.get("/api/viral_tweets")
def get_viral_tweets():
    # TODO: Return list of top viral tweets
    return {"tweets": []}

@app.get("/api/emerging_keywords")
def get_emerging_keywords():
    # TODO: Return list/word cloud of keywords
    return {"keywords": []}

@app.get("/api/tracked_accounts")
def get_tracked_accounts():
    # TODO: Return leaderboard of tracked accounts
    return {"accounts": []}

@app.get("/api/activity_log")
def get_activity_log():
    # TODO: Return system activity log
    return {"log": []} 