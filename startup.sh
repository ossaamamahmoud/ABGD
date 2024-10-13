#!/bin/bash

# Set the environment variable for the upload folder
export UPLOAD_FOLDER=uploads/

# Ensure the upload folder exists
if [ ! -d "$UPLOAD_FOLDER" ]; then
  mkdir -p "$UPLOAD_FOLDER"
fi

# Start the Gunicorn server, binding it to 0.0.0.0 on port 8000, and point to the app object in app.py
gunicorn --bind 0.0.0.0:8000 app:app
