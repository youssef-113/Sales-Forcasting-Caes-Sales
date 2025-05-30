#!/bin/bash
# This script sets up the virtual environment and installs dependencies.

echo "Creating virtual environment 'venv'..."
python3 -m venv venv
if [ $? -ne 0 ]; then
    echo "Failed to create virtual environment. Please ensure python3 and venv are installed."
    exit 1
fi

echo "Activating virtual environment..."
# The activation command differs between OS environments.
# This script provides a common way to inform the user.
echo "On Linux/macOS, run: source venv/bin/activate"
echo "On Windows (Git Bash or similar), run: source venv/Scripts/activate"
echo "On Windows (Command Prompt), run: venv\\Scripts\\activate.bat"
echo "On Windows (PowerShell), run: venv\\Scripts\\Activate.ps1"

echo "Attempting to install dependencies from requirements.txt into 'venv'..."
# Try to find the pip executable in the venv to install packages directly
VENV_PIP="venv/bin/pip"
if [[ "$OSTYPE" == "msys" || "$OSTYPE" == "cygwin" || "$OSTYPE" == "win32" ]]; then
    VENV_PIP="venv/Scripts/pip"
fi

if [ -f "$VENV_PIP" ]; then
    $VENV_PIP install -r requirements.txt
else
    echo "Could not find pip in venv. Please activate the venv manually and then run: pip install -r requirements.txt"
    exit 1
fi

if [ $? -ne 0 ]; then
    echo "Failed to install dependencies. Please check requirements.txt and ensure pip is working correctly after activating the environment."
    exit 1
fi

echo "Setup complete. Activate your virtual environment to run the project."
