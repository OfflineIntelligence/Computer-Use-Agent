#!/usr/bin/env python3
"""
Computer Use Agent - Main Entry Point
=====================================

This is the main entry point for the computer use agent system.
The agent connects to a locally hosted Rust LLM server and performs
computer automation tasks through multimodal interaction.

Usage:
    python main.py                    # Start interactive chat interface
    python main.py --test             # Run quick system test
    python main.py --command "task"   # Execute single command
"""

import sys
import os
import argparse
import logging
from typing import Optional

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

# Import with relative paths to avoid module issues
sys.path.insert(0, os.path.join(current_dir, 'Agent_hub'))
sys.path.insert(0, os.path.join(current_dir, 'Core'))
sys.path.insert(0, os.path.join(current_dir, 'Memory'))
sys.path.insert(0, os.path.join(current_dir, 'Utils'))

from Agent1 import Agent
from Config import Config

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('agent.log')
    ]
)

logger = logging.getLogger("Main")


def run_interactive_mode():
    """Run the interactive chat interface"""
    logger.info("Starting interactive mode...")
    try:
        agent = Agent()
        agent.chat_interface()
    except KeyboardInterrupt:
        print("\n\n🛑 Shutdown requested by user")
    except Exception as e:
        logger.error(f"Interactive mode failed: {e}")
        print(f"❌ Error: {e}")


def run_single_command(command: str):
    """Execute a single command"""
    logger.info(f"Executing command: {command}")
    try:
        agent = Agent()
        info, actions = agent.predict(command)
        
        print(f"\n✅ Command Execution Complete!")
        print(f"📋 Final Status: {info.get('status', 'Unknown')}")
        print(f"🎯 Actions Taken: {len(actions)}")
        print(f"  Details: {actions}")
        
        # Show any errors
        if 'error' in info:
            print(f"⚠️  Errors: {info['error']}")
            
    except Exception as e:
        logger.error(f"Command execution failed: {e}")
        print(f"❌ Execution failed: {e}")


def run_system_test():
    """Run system diagnostics and tests"""
    print("🧪 Running System Diagnostics...")
    print("=" * 50)
    
    # Test 1: Configuration loading
    print("1. Testing configuration loading...")
    try:
        rust_config = Config.get_rust_config()
        agent_config = Config.get_agent_config()
        print(f"   ✅ Rust server: {rust_config['base_url']}")
        print(f"   ✅ Platform: {agent_config['platform']}")
        print(f"   ✅ Safety mode: {agent_config['safety_mode']}")
    except Exception as e:
        print(f"   ❌ Configuration error: {e}")
        return False
    
    # Test 2: Agent initialization
    print("\n2. Testing agent initialization...")
    try:
        agent = Agent()
        print(f"   ✅ Agent created successfully")
        print(f"   ✅ Platform: {agent.platform}")
        print(f"   ✅ Model: {agent.engine_parameters.get('model', 'Unknown')}")
    except Exception as e:
        print(f"   ❌ Agent initialization failed: {e}")
        return False
    
    # Test 3: Backend connectivity
    print("\n3. Testing backend connectivity...")
    try:
        from Entity.Core.Core_engine import RustServer
        engine = RustServer(**agent.engine_parameters)
        print(f"   ✅ Backend connection established")
    except Exception as e:
        print(f"   ⚠️  Backend connection warning: {e}")
        print("      Note: This may be normal if Rust server isn't running yet")
    
    # Test 4: OCR availability
    print("\n4. Testing OCR capabilities...")
    try:
        import pytesseract
        tesseract_path = Config.TESSERACT_PATH
        if tesseract_path and os.path.exists(tesseract_path):
            pytesseract.pytesseract.tesseract_cmd = tesseract_path
            print(f"   ✅ Tesseract OCR configured: {tesseract_path}")
        else:
            print(f"   ⚠️  Using system Tesseract path")
    except Exception as e:
        print(f"   ❌ OCR setup failed: {e}")
    
    print("\n" + "=" * 50)
    print("✅ System diagnostics completed!")
    print("\nNext steps:")
    print("1. Ensure your Rust LLM server is running at http://127.0.0.1:8000")
    print("2. Update .env file with your model configuration")
    print("3. Run 'python main.py' for interactive mode")
    
    return True


def main():
    """Main entry point"""
    parser = argparse.ArgumentParser(
        description="Computer Use Agent - Automate computer tasks with AI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                          # Interactive chat mode
  python main.py --test                   # Run system diagnostics  
  python main.py --command "open notepad" # Execute single command
        """
    )
    
    parser.add_argument(
        '--test', 
        action='store_true',
        help='Run system diagnostics and tests'
    )
    
    parser.add_argument(
        '--command', 
        type=str,
        help='Execute a single command directly'
    )
    
    parser.add_argument(
        '--verbose', 
        '-v',
        action='store_true',
        help='Enable verbose logging'
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        if args.test:
            success = run_system_test()
            sys.exit(0 if success else 1)
        elif args.command:
            run_single_command(args.command)
        else:
            run_interactive_mode()
            
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        sys.exit(0)
    except Exception as e:
        logger.error(f"Fatal error: {e}", exc_info=True)
        print(f"❌ Fatal error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()