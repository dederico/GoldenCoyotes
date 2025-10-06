#!/usr/bin/env python3
"""
Simplified Main Entry Point for Business Dealer Intelligence System
A simplified version that starts the Flask service without complex AI initialization
"""

import os
import logging
from datetime import datetime
import click
from flask import Flask

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

def create_simple_app():
    """Create Flask app with minimal configuration"""
    
    # Simple config class
    class SimpleConfig:
        def __init__(self):
            self.debug = True
            self.environment = "development"
    
    # Simple components class  
    class SimpleComponents:
        def __init__(self):
            pass
    
    # Import and create app
    from api.main_api import create_app
    
    config = SimpleConfig()
    components = SimpleComponents()
    
    return create_app(config, components)

@click.command()
@click.option('--port', default=8080, help='Port to run the service on')
@click.option('--debug', is_flag=True, help='Run in debug mode')
@click.option('--host', default='0.0.0.0', help='Host to bind to')
def main(port, debug, host):
    """Start the Business Dealer Intelligence Service (Simplified)"""
    
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           Business Dealer Intelligence System                ║
    ║                     (Simplified Mode)                       ║
    ║  🧠 AI-Powered Business Intelligence & Recommendations       ║
    ║  📊 Real-time Analytics & Insights                          ║
    ║  🔔 Smart Notifications & Context Analysis                  ║
    ║  🤖 Machine Learning Models & Predictions                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    logger.info("🚀 Starting Business Dealer Intelligence Service (Simplified)")
    logger.info(f"📊 Service starting at {datetime.now()}")
    
    try:
        # Create Flask app
        app = create_simple_app()
        
        logger.info(f"✅ Service initialized successfully")
        logger.info(f"🌐 Starting Flask server on {host}:{port}")
        logger.info(f"📋 Access Dashboard: http://localhost:{port}/dashboard")
        logger.info(f"🔍 Health Check: http://localhost:{port}/health")
        logger.info(f"📡 API Status: http://localhost:{port}/api/intelligence/status")
        
        # Start Flask development server
        app.run(
            host=host,
            port=port,
            debug=debug,
            use_reloader=False  # Disable reloader to avoid duplicate startup
        )
        
    except KeyboardInterrupt:
        logger.info("🛑 Service stopped by user")
    except Exception as e:
        logger.error(f"❌ Failed to start service: {e}")
        raise

if __name__ == "__main__":
    main()