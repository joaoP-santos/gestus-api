"""
Configuration settings for the Gestus API
"""
import os

# Flask settings
FLASK_ENV = os.getenv("FLASK_ENV", "production")
PORT = int(os.getenv("PORT", 5000))

# Model settings
MODEL_PATH = os.getenv("MODEL_PATH", "best_model.pth")

# Supabase settings (replace with your actual credentials)
SUPABASE_URL = os.getenv("SUPABASE_URL", "https://bctfiwwbascthdbttavj.supabase.co")
SUPABASE_KEY = os.getenv("SUPABASE_KEY", "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6ImJjdGZpd3diYXNjdGhkYnR0YXZqIiwicm9sZSI6InNlcnZpY2Vfcm9sZSIsImlhdCI6MTc0NjY0MTA5NiwiZXhwIjoyMDYyMjE3MDk2fQ.foFL5AtIBhnitzKaot2xNJuR4mwVO046ttktYIiwXTs")
SUPABASE_BUCKET = os.getenv("SUPABASE_BUCKET", "libras-dataset")

# Available signs
SIGNS = [
    "acontecer", "aluno", "amarelo", "america", 
    "aproveitar", "bala", "banco", "banheiro", 
    "barulho", "cinco", "conhecer", "espelho", 
    "esquina", "filho", "maca", "medo", "ruim", 
    "sapo", "vacina", "vontade"
]