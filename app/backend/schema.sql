CREATE TABLE IF NOT EXISTS tweets (
  id TEXT PRIMARY KEY,
  text TEXT,
  author_id TEXT,
  author_handle TEXT,
  created_at DATETIME,
  retweets INTEGER,
  likes INTEGER,
  replies INTEGER,
  quotes INTEGER,
  followers INTEGER,
  score REAL,
  velocity REAL,
  category TEXT
);

CREATE TABLE IF NOT EXISTS accounts (
  handle TEXT PRIMARY KEY,
  last_checked DATETIME,
  avg_velocity REAL,
  viral_count INTEGER,
  score REAL
);

CREATE TABLE IF NOT EXISTS keywords (
  term TEXT PRIMARY KEY,
  frequency INTEGER,
  avg_score REAL,
  last_seen DATETIME
);

CREATE TABLE IF NOT EXISTS mentions (
  source TEXT,
  target TEXT,
  count INTEGER,
  PRIMARY KEY (source, target)
);

CREATE TABLE IF NOT EXISTS activity_log (
  id INTEGER PRIMARY KEY AUTOINCREMENT,
  timestamp DATETIME,
  event_type TEXT,
  details TEXT
); 