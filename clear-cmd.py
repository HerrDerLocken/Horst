import requests
import os
from dotenv import load_dotenv

load_dotenv()
TOKEN = os.getenv("TOKEN")

# Get your bot's application ID from the Discord Developer Portal
APPLICATION_ID = "YOUR_APPLICATION_ID_HERE"

headers = {"Authorization": f"Bot {TOKEN}"}

# Wipe all global commands
r = requests.put(
    f"https://discord.com/api/v10/applications/{APPLICATION_ID}/commands",
    headers=headers,
    json=[]
)
print("Global commands cleared:", r.status_code)