#!/usr/bin/env python3
"""
Business Dealer Intelligence System - Main Entry Point
Startup script for the complete Business Dealer Intelligence Service

This script:
- Initializes all system components
- Sets up database connections
- Starts the web API server
- Provides health checking and monitoring
- Handles graceful shutdown

Usage:
    python main.py                  # Start with default settings
    python main.py --port 8080      # Start on specific port
    python main.py --debug          # Start in debug mode
    python main.py --help           # Show help
"""

import asyncio
import logging
import signal
import sys
import os
from datetime import datetime
from pathlib import Path
import click

# Add the project root to Python path
sys.path.insert(0, str(Path(__file__).parent))

from config.intelligence_config import IntelligenceConfig
from config.ml_config import MLConfig
from intelligence.master_engine import MasterIntelligenceEngine
from analytics.metrics_calculator import MetricsCalculator
from analytics.insight_generator import InsightGenerator
from notification.smart_prioritizer import SmartPrioritizer
from notification.context_analyzer import ContextAnalyzer
from database.intelligence_schema import IntelligenceSchema

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('business_intelligence.log')
    ]
)
logger = logging.getLogger(__name__)


class BusinessIntelligenceService:
    """Main service class for Business Dealer Intelligence System"""
    
    def __init__(self, config_override=None):
        """Initialize the service"""
        self.config = config_override or self._load_config()
        self.ml_config = MLConfig.from_env()
        self.running = False
        self.components = {}
        
        logger.info("🚀 Initializing Business Dealer Intelligence Service")
        logger.info(f"📊 Service starting at {datetime.now()}")
        
    def _load_config(self):
        """Load configuration with environment variable fallbacks"""
        try:
            return IntelligenceConfig.from_env()
        except Exception as e:
            logger.warning(f"⚠️ Could not load full config: {e}")
            logger.info("📝 Using minimal configuration for demo/testing")
            
            # Create minimal config for demo
            from config.intelligence_config import (
                IntelligenceConfig, OpenAIConfig, RedisConfig, 
                DatabaseConfig, BehaviorAnalysisConfig, NotificationConfig, AnalyticsConfig
            )
            
            return IntelligenceConfig(
                environment="development",
                debug=True,
                log_level=os.getenv("LOG_LEVEL", "INFO"),
                openai=OpenAIConfig(
                    api_key=os.getenv("OPENAI_API_KEY", "demo_key"),
                    chat_model="gpt-4o-mini",
                    embedding_model="text-embedding-3-large",
                    embedding_dimensions=3072,
                    timeout=30,
                    max_retries=3,
                    temperature=0.1,
                    max_tokens=4096
                ),
                redis=RedisConfig(
                    url=os.getenv("REDIS_URL", "redis://localhost:6379"),
                    default_ttl=300,
                    embedding_cache_ttl=3600,
                    analytics_cache_ttl=600
                ),
                database=DatabaseConfig(
                    intelligence_db_path=os.getenv("INTELLIGENCE_DB_PATH", "intelligence.db")
                ),
                behavior_analysis=BehaviorAnalysisConfig(),
                notification=NotificationConfig(),
                analytics=AnalyticsConfig()
            )
    
    async def initialize_components(self):
        """Initialize all service components"""
        try:
            logger.info("🔧 Initializing system components...")
            
            # Setup database
            logger.info("📁 Setting up database...")
            schema = IntelligenceSchema(self.config.database.intelligence_db_path)
            await asyncio.get_event_loop().run_in_executor(None, schema.create_database)
            
            # Initialize core components
            logger.info("🧠 Initializing Master Intelligence Engine...")
            self.components['master_engine'] = MasterIntelligenceEngine(
                self.config
            )
            
            logger.info("📊 Initializing Analytics Components...")
            self.components['metrics_calculator'] = MetricsCalculator(
                self.config, self.ml_config
            )
            self.components['insight_generator'] = InsightGenerator(
                self.config, self.ml_config
            )
            
            logger.info("🔔 Initializing Notification System...")
            self.components['smart_prioritizer'] = SmartPrioritizer(
                self.config, self.ml_config
            )
            self.components['context_analyzer'] = ContextAnalyzer(
                self.config, self.ml_config
            )
            
            logger.info("✅ All components initialized successfully!")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to initialize components: {e}")
            return False
    
    async def start_service(self, port=8080, debug=False):
        """Start the service"""
        try:
            logger.info(f"🚀 Starting Business Dealer Intelligence Service on port {port}")
            
            # Initialize components
            if not await self.initialize_components():
                logger.error("❌ Failed to initialize components")
                return False
            
            # Start web API server
            from api.main_api import create_app
            app = create_app(self.config, self.components)
            
            # Configure Flask app
            if debug:
                app.config['DEBUG'] = True
                app.config['TESTING'] = True
            
            logger.info("🌐 Starting Flask API server...")
            
            # In production, you'd use a proper WSGI server like Gunicorn
            # For demo purposes, we'll use Flask's development server
            self.running = True
            
            # Setup signal handlers for graceful shutdown
            def signal_handler(signum, frame):
                logger.info(f"📊 Received signal {signum}, shutting down...")
                asyncio.create_task(self.shutdown())
            
            signal.signal(signal.SIGINT, signal_handler)
            signal.signal(signal.SIGTERM, signal_handler)
            
            # Start the Flask app
            logger.info("✅ Service started successfully!")
            logger.info(f"📱 API available at: http://localhost:{port}")
            logger.info(f"📊 Dashboard available at: http://localhost:{port}/dashboard")
            logger.info(f"🔍 Health check at: http://localhost:{port}/health")
            
            # Run the Flask app
            app.run(host='0.0.0.0', port=port, debug=debug, use_reloader=False)
            
        except Exception as e:
            logger.error(f"❌ Failed to start service: {e}")
            return False
    
    async def shutdown(self):
        """Graceful shutdown of the service"""
        logger.info("🛑 Initiating graceful shutdown...")
        
        self.running = False
        
        # Close component connections
        for name, component in self.components.items():
            try:
                if hasattr(component, 'close'):
                    await component.close()
                logger.info(f"✅ Closed {name}")
            except Exception as e:
                logger.warning(f"⚠️ Error closing {name}: {e}")
        
        logger.info("✅ Shutdown complete")
    
    def get_service_status(self):
        """Get current service status"""
        status = {
            "service": "Business Dealer Intelligence",
            "version": "1.0.0",
            "status": "running" if self.running else "stopped",
            "started_at": datetime.now().isoformat(),
            "components": {}
        }
        
        for name, component in self.components.items():
            try:
                if hasattr(component, 'get_status'):
                    status["components"][name] = component.get_status()
                else:
                    status["components"][name] = {"status": "initialized"}
            except Exception as e:
                status["components"][name] = {"status": "error", "error": str(e)}
        
        return status


# CLI interface
@click.command()
@click.option('--port', default=8080, help='Port to run the service on')
@click.option('--debug', is_flag=True, help='Run in debug mode')
@click.option('--setup-only', is_flag=True, help='Only setup database and exit')
@click.option('--health-check', is_flag=True, help='Check service health and exit')
def main(port, debug, setup_only, health_check):
    """Start the Business Dealer Intelligence Service"""
    
    # Print banner
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║           Business Dealer Intelligence System                ║
    ║                                                              ║
    ║  🧠 AI-Powered Business Intelligence & Recommendations       ║
    ║  📊 Real-time Analytics & Insights                          ║
    ║  🔔 Smart Notifications & Context Analysis                  ║
    ║  🤖 Machine Learning Models & Predictions                   ║
    ║                                                              ║
    ╚══════════════════════════════════════════════════════════════╝
    """)
    
    # Create service instance
    service = BusinessIntelligenceService()
    
    if setup_only:
        logger.info("🔧 Running setup only...")
        asyncio.run(service.initialize_components())
        logger.info("✅ Setup complete!")
        return
    
    if health_check:
        logger.info("🔍 Running health check...")
        status = service.get_service_status()
        print(f"Service Status: {status['status']}")
        return
    
    # Start the service
    try:
        asyncio.run(service.start_service(port=port, debug=debug))
    except KeyboardInterrupt:
        logger.info("🛑 Service interrupted by user")
    except Exception as e:
        logger.error(f"❌ Service error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()