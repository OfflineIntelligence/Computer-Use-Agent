#!/usr/bin/env python3
"""
Usage Examples for Computer Use Agent
=====================================

This script demonstrates various ways to use the computer use agent
for different automation tasks.
"""

import sys
import os
import time

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
if current_dir not in sys.path:
    sys.path.insert(0, current_dir)

sys.path.insert(0, os.path.join(current_dir, 'Agent_hub'))
sys.path.insert(0, os.path.join(current_dir, 'Core'))
sys.path.insert(0, os.path.join(current_dir, 'Memory'))
sys.path.insert(0, os.path.join(current_dir, 'Utils'))

from Agent1 import Agent


def example_basic_commands():
    """Demonstrate basic computer automation commands"""
    print("=== Basic Commands Demo ===")
    
    agent = Agent()
    
    # Simple application opening
    commands = [
        "open calculator",
        "open notepad",
        "open chrome"
    ]
    
    for cmd in commands:
        print(f"\n🎯 Executing: {cmd}")
        try:
            info, actions = agent.predict(cmd)
            print(f"✅ Result: {len(actions)} actions executed")
            time.sleep(2)  # Brief pause between commands
        except Exception as e:
            print(f"❌ Failed: {e}")


def example_text_operations():
    """Demonstrate text input and editing operations"""
    print("\n=== Text Operations Demo ===")
    
    agent = Agent()
    
    # Text manipulation tasks
    commands = [
        "open notepad and type Hello World",
        "type today's date in the document",
        "select all text and copy it"
    ]
    
    for cmd in commands:
        print(f"\n🎯 Executing: {cmd}")
        try:
            info, actions = agent.predict(cmd)
            print(f"✅ Result: {len(actions)} actions executed")
            time.sleep(3)  # Longer pause for text operations
        except Exception as e:
            print(f"❌ Failed: {e}")


def example_navigation():
    """Demonstrate navigation and browsing tasks"""
    print("\n=== Navigation Demo ===")
    
    agent = Agent()
    
    # Navigation tasks
    commands = [
        "open chrome and go to google.com",
        "search for artificial intelligence",
        "click the first search result"
    ]
    
    for cmd in commands:
        print(f"\n🎯 Executing: {cmd}")
        try:
            info, actions = agent.predict(cmd)
            print(f"✅ Result: {len(actions)} actions executed")
            time.sleep(4)  # Longer pause for web navigation
        except Exception as e:
            print(f"❌ Failed: {e}")


def example_complex_workflow():
    """Demonstrate complex multi-step workflows"""
    print("\n=== Complex Workflow Demo ===")
    
    agent = Agent()
    
    # Complex task combining multiple operations
    complex_task = (
        "open word processor, "
        "type a professional email template, "
        "save the document as 'business_email_template.docx', "
        "and close the application"
    )
    
    print(f"\n🎯 Executing complex task:")
    print(f"   {complex_task}")
    
    try:
        info, actions = agent.predict(complex_task)
        print(f"✅ Completed: {len(actions)} actions executed")
        print(f"📋 Final status: {info.get('status', 'Unknown')}")
    except Exception as e:
        print(f"❌ Failed: {e}")


def example_safety_features():
    """Demonstrate safety features and confirmation prompts"""
    print("\n=== Safety Features Demo ===")
    
    agent = Agent()
    
    # Destructive actions that require confirmation
    destructive_commands = [
        "type password123 in the active window",  # Will require confirmation
        "press ctrl+alt+delete",                  # System-level action
        "delete all files in documents folder"    # Would require confirmation
    ]
    
    print("The following commands demonstrate safety features:")
    print("(Some may require manual confirmation)")
    
    for cmd in destructive_commands[:1]:  # Only show first one for demo
        print(f"\n🎯 Command: {cmd}")
        print("   (This would trigger safety confirmation in 'confirm' mode)")


def main():
    """Run all examples"""
    print("🤖 Computer Use Agent - Usage Examples")
    print("=" * 50)
    
    try:
        # Run examples
        example_basic_commands()
        example_text_operations()
        example_navigation()
        example_complex_workflow()
        example_safety_features()
        
        print("\n" + "=" * 50)
        print("✅ All examples completed!")
        print("\n💡 Tips:")
        print("• Start with simple commands to test functionality")
        print("• Use verbose mode (-v) for debugging")
        print("• Check .env configuration for optimal performance")
        print("• Review safety settings before running destructive actions")
        
    except KeyboardInterrupt:
        print("\n\n🛑 Examples interrupted by user")
    except Exception as e:
        print(f"\n❌ Examples failed: {e}")


if __name__ == "__main__":
    main()