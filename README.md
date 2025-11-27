# AI Codebench

A comprehensive CLI-based AI assistant capable of interacting with multiple AI providers including Anthropic (Claude), DeepSeek, Google Gemini, Moonshot (Kimi), and OpenRouter. Designed for coding, learning, and writing tasks with rich terminal output and extensive configuration options.

## Features

- **Multi-Provider Support:** Seamlessly switch between Claude, DeepSeek, Gemini, Kimi, and OpenRouter.
- **Task-Specific Modes:** Optimized prompts and models for:
  - `code`: Algorithm analysis and coding assistance.
  - `learn`: Step-by-step concept explanations.
  - `write`: Polishing and editing text while preserving voice.
- **Rich CLI Interface:** Beautiful terminal output with Markdown rendering, tables, and panels (powered by `rich`).
- **Conversation Management:** Automatic history tracking with context caching support.
- **Response Recording:** Option to save Q&A pairs to local Markdown files automatically.
- **Background Processing:** Support for asynchronous saving of responses to avoid blocking the user interface.
- **Searchable Archive:** Built-in archiving system with search functionality to retrieve past answers.
- **Flexible Configuration:** Configure default models, provider fallbacks, and routing logic via YAML or environment variables.

## Installation

### Prerequisites
- Python 3.9 or higher

### Using pip
```bash
pip install -e .
```

### Using PDM
```bash
pdm install
```

## Usage

Run the CLI application:

```bash
poetry env activate && poetry run python main.py
```

### Command Line Arguments

You can pre-configure the session using flags:

- `-p, --provider [claude|deepseek|gemini|openrouter|kimi]`: Set initial provider.
- `-t, --task [code|learn|write]`: Set initial task type.
- `-m, --mode [sync|async]`: Set streaming mode.
- `-M, --multi [on|off]`: Enable multi-line input mode.
- `-r, --record [on|off]`: Enable response recording to files.
- `-b, --background [on|off]`: Enable background saving (requires record=on).

**Example:**
```bash
poetry run python main.py --provider deepseek --task code --record on
```

### Interactive Commands

Once inside the CLI, you can use the following slash commands:

- `/help`: Show available commands.
- `/provider [name]`: Switch active provider.
- `/model`: List available models for the current provider.
- `/task [code|learn|write]`: Switch task context.
- `/stat`: Show current session statistics (tokens, turns).
- `/mode [sync|async]`: Toggle between synchronous and asynchronous streaming.
- `/multi [on|off]`: Toggle multi-line input mode (useful for pasting code).
- `/record [on|off]`: Toggle saving of responses to local files.
- `/background [on|off]`: Toggle background saving of answers.
- `/search [query]`: Search through archived answers and restore them.
- `/exit`: Quit the application.

## License

MIT