#!/usr/bin/env bash
set -e  # Exit on error

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

echo -e "${GREEN}Setting up Generative Agents...${NC}"

# Check Python version
if ! command -v python3.12 &> /dev/null; then
    echo -e "${RED}Python 3.12 is required but not found.${NC}"
    echo -e "${YELLOW}Please install Python 3.12 and try again.${NC}"
    exit 1
fi

# Create virtual environment if it doesn't exist
if [ ! -d ".venv" ]; then
    echo -e "${GREEN}Creating virtual environment...${NC}"
    python3.12 -m venv .venv
fi

# Activate virtual environment
echo -e "${GREEN}Activating virtual environment...${NC}"
source .venv/bin/activate

# Upgrade pip and install dependencies
echo -e "${GREEN}Installing dependencies...${NC}"
python -m pip install --upgrade pip
pip install -r requirements.txt

# Check environment configuration
if [ "${MODEL_TYPE}" != "ollama_"* ]; then
    API_KEY_VAR="$(printenv | grep -o '^[^=]*' | grep '_API_KEY$' | head -n 1)"
    if [ -z "${API_KEY_VAR}" ]; then
        echo -e "${RED}Warning: No API key configured for cloud models.${NC}"
        echo -e "${YELLOW}Please set appropriate API key in environment.${NC}"
    fi
fi

if [ "${MODEL_TYPE}" == "ollama_"* ]; then
    echo -e "${GREEN}Checking Ollama installation...${NC}"
    if ! command -v ollama &> /dev/null; then
        echo -e "${RED}Ollama is required but not found.${NC}"
        echo -e "${YELLOW}Please install Ollama from https://ollama.ai${NC}"
        exit 1
    fi
fi

# Start environment server
echo -e "${GREEN}Starting environment server...${NC}"
cd environment/frontend_server
python manage.py runserver &
FRONTEND_PID=$!

# Give the frontend server time to start
sleep 2

# Start simulation server
echo -e "${GREEN}Starting simulation server...${NC}"
cd ../../reverie/backend_server
python reverie.py &
BACKEND_PID=$!

# Trap Ctrl+C to properly shut down both servers
trap 'kill $FRONTEND_PID $BACKEND_PID; exit' INT

# Wait for either process to exit
wait -n
kill $FRONTEND_PID $BACKEND_PID 2>/dev/null
