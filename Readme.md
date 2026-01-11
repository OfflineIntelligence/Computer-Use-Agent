# Computer Use Agent (CUA)

A sophisticated computer automation agent that uses a locally-hosted LLM to perform computer tasks through multimodal interaction.

## 🏗️ Architecture Overview

The system follows a modular architecture with clear separation of concerns:

```
Entity/
├── Core/                 # Core LLM and engine components
│   ├── Core_engine.py    # Rust server communication layer
│   ├── Multimodal_llm.py # Multimodal LLM wrapper
│   └── Component.py      # Base component framework
├── Agent_hub/           # Agent coordination and execution
│   ├── Action_board.py   # Planning and task decomposition
│   ├── Execution_console.py # Action execution and UI interaction
│   └── Agent1.py         # Main agent orchestration
├── Memory/              # State and memory management
│   └── Operational_memory.py # Prompt construction and caching
├── Utils/               # Utilities and configuration
│   ├── Common_utilities.py # Helper functions and data classes
│   └── Config.py        # Configuration management
├── main.py              # Main entry point
└── .env                 # Environment configuration
```

## 🚀 Quick Start

### Prerequisites
1. Python 3.8+
2. Locally hosted Rust LLM server (running at `http://127.0.0.1:8000`)
3. Tesseract OCR (optional but recommended)

### Installation
```bash
pip install -r requirements.txt
```

### Configuration
Edit the `.env` file with your settings:
```env
# Rust Server Configuration
RUST_SERVER_URL=http://127.0.0.1:8000
RUST_SERVER_MODEL=your-model-name
RUST_SERVER_TIMEOUT=600
RUST_SERVER_MAX_TOKENS=4096

# Agent Configuration
AGENT_PLATFORM=windows
AGENT_SAFETY_MODE=confirm  # confirm|execute|dry-run
```

### Usage

#### Interactive Mode
```bash
python main.py
```

#### Single Command Execution
```bash
python main.py --command "open notepad and type hello world"
```

#### System Diagnostics
```bash
python main.py --test
```

## 🧠 Core Components

### 1. Core Engine (`Core_engine.py`)
Handles communication with the Rust LLM server:
- HTTP/REST API integration
- Streaming response support
- Retry mechanisms with exponential backoff
- Error handling and timeouts

### 2. Multimodal LLM (`Multimodal_llm.py`)
Manages multimodal inputs (text + images):
- Base64 image encoding/decoding
- Conversation history management
- Format conversion for Rust server compatibility

### 3. Action Board (`Action_board.py`)
Pure planning component:
- Task decomposition into executable steps
- Plan generation using LLM reasoning
- Step-by-step execution coordination
- Replanning capabilities for failed steps

### 4. Execution Console (`Execution_console.py`)
UI automation and action execution:
- PyAutoGUI integration for desktop automation
- OCR-based coordinate detection (Pytesseract)
- Visual feedback analysis
- Safety mechanisms and confirmation prompts

### 5. Agent Orchestrator (`Agent1.py`)
Main coordination layer:
- Iterative task execution loop
- State management across multiple steps
- Early stopping criteria
- Chat interface for user interaction

## 🔧 Configuration Options

### Safety Modes
- `confirm`: Requires user confirmation for destructive actions
- `execute`: Executes all actions without confirmation
- `dry-run`: Shows planned actions without execution

### Platform Support
Currently supports Windows (Linux/macOS support can be added by extending platform-specific commands)

### OCR Configuration
Configure Tesseract path in `.env`:
```env
TESSERACT_PATH=C:\Program Files\Tesseract-OCR\tesseract.exe
```

## 📊 System Capabilities

### Supported Actions
- **Application Control**: Open/close applications
- **UI Interaction**: Click, type, scroll, press keys
- **Navigation**: Browser navigation, menu selection
- **System Commands**: Hotkeys, file operations
- **Visual Analysis**: Screenshot capture and analysis

### Planning Features
- Multi-step task decomposition
- Adaptive replanning for failures
- Progress tracking and monitoring
- Context-aware decision making

## 🛡️ Safety Features

### Destructive Action Protection
The system requires confirmation for potentially harmful actions:
- Text typing in applications
- Keyboard shortcuts execution
- Drag-and-drop operations

### Error Recovery
- Automatic retry mechanisms
- Graceful degradation on failures
- Detailed error reporting
- Visual feedback analysis for verification

## 🎯 Example Usage

### Simple Task
```
> open calculator
🚀 Executing: 'open calculator'
⏳ Thinking...

✅ Execution Complete!
📋 Final Plan: Open Windows Calculator
🎯 Actions Taken: 1
  Details: ['Attempted to open: calculator']
```

### Complex Multi-step Task
```
> open notepad, type "Hello World", save as "test.txt", and close
🚀 Executing: 'open notepad, type "Hello World", save as "test.txt", and close'
⏳ Thinking...

✅ Execution Complete!
📋 Final Plan: Complete document workflow
🎯 Actions Taken: 5
  Details: ['Attempted to open: notepad', 'Action completed successfully', ...]
```

## 📈 Performance Optimization

### Memory Management
- LRU caching for prompts and trajectories
- Configurable context window sizes
- Automatic cleanup of stale conversations

### Response Streaming
Enable streaming for faster response times:
```env
AGENT_STREAM_RESPONSES=true
```

## 🐛 Troubleshooting

### Common Issues

1. **Connection Refused**: Ensure Rust server is running at configured URL
2. **OCR Not Working**: Verify Tesseract installation and path configuration
3. **Permission Denied**: Run with appropriate privileges for system automation
4. **Coordinate Detection Failures**: Improve lighting conditions or adjust confidence thresholds

### Debugging
Enable verbose logging:
```bash
python main.py --verbose
```

## 🤝 Contributing

### Development Setup
1. Fork the repository
2. Create feature branch
3. Make changes
4. Test thoroughly
5. Submit pull request

### Extending Functionality
- Add new action types in `Execution_console.py`
- Extend planning capabilities in `Action_board.py`
- Add platform support in respective modules

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- Built on top of PyAutoGUI for desktop automation
- Uses Tesseract OCR for visual element detection
- Inspired by OSWorld and computer use agent research