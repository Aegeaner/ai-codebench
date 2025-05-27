# AI Codebench

A command-line interface for interacting with multiple AI providers including OpenAI, Anthropic, and Google Gemini.

## Features

- Unified interface for multiple AI providers
- CLI-based interaction
- Support for OpenAI, Anthropic, and Google Gemini APIs
- Rich output formatting
- Configuration management

## Installation

```bash
# Using pip
pip install -e .

# Using pdm
pdm install
```

## Usage

```bash
# Run the CLI
python -m ai_codebench.cli
```

The tool supports multiple providers which can be configured via `config.yaml`.

## Configuration

Create a `config.yaml` file with your API keys:

```yaml
providers:
  openai:
    api_key: "your-openai-key"
  anthropic:
    api_key: "your-anthropic-key"
  google:
    api_key: "your-google-key"
```

## Supported Providers

- OpenAI (via `openai_provider.py`)
- Anthropic (via `claude_provider.py`)
- Google Gemini (via `gemini_provider.py`)

## License

MIT
