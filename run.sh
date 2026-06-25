#!/bin/bash
# NRL Sentinel - Entry Point (Bash)
# For Linux and macOS users

echo ""
echo "╔════════════════════════════════════════════════════════════════╗"
echo "║                                                                ║"
echo "║  🛢️   NRL SENTINEL — AI Decision Intelligence System           ║"
echo "║      Numaligarh Refinery Limited | NREP Risk Analytics        ║"
echo "║                                                                ║"
echo "╚════════════════════════════════════════════════════════════════╝"
echo ""

# Check if virtual environment exists
if [ ! -d "venv" ] && [ ! -d ".venv" ]; then
    echo "⚠️  Virtual environment not found!"
    echo "   Creating one with: python -m venv venv"
    python -m venv venv
    echo "✅ Virtual environment created"
    echo ""
fi

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
fi

# Check if requirements are installed
echo "📦 Checking dependencies..."
python -c "import streamlit; import pandas; import plotly" 2>/dev/null
if [ $? -ne 0 ]; then
    echo "⚠️  Installing dependencies from requirements.txt..."
    pip install -r nrl_sentinel/requirements.txt --quiet
    echo "✅ Dependencies installed"
    echo ""
fi

# Parse arguments
if [ "$1" == "--data-only" ]; then
    echo "📊 Generating datasets..."
    cd nrl_sentinel
    python -c "from data_collector import fetch_crude_and_product_prices, generate_ne_india_demand, generate_sensor_data, generate_project_data; fetch_crude_and_product_prices(); generate_ne_india_demand(); generate_sensor_data(); generate_project_data(); print('\n✅ All datasets generated successfully!')"
    cd ..
elif [ "$1" == "--help" ] || [ "$1" == "-h" ]; then
    echo "Usage: bash run.sh [OPTION]"
    echo ""
    echo "OPTIONS:"
    echo "  (none)          Launch Streamlit dashboard"
    echo "  --data-only     Generate/update datasets only"
    echo "  --help, -h      Show this help message"
    echo ""
else
    # Launch Streamlit dashboard
    echo "🚀 Launching NRL Sentinel Dashboard..."
    echo ""
    echo "   🌐 Opening at: http://localhost:8501"
    echo "   ⏹️  Press Ctrl+C to stop"
    echo ""
    python -m streamlit run nrl_sentinel/app.py --logger.level=info
fi
