# Custom LLM Configuration for Cline

## Overview
Cline supports adding custom LLM endpoints that are compatible with the OpenAI API format. This allows you to use your own LLM running on a cluster.

## Configuration Methods

### 1. VS Code Settings UI (Recommended)
1. Open VS Code Settings:
   - Press `Ctrl+,` (or `Cmd+,` on macOS)
   - Or go to File → Preferences → Settings

2. Search for "Cline" in the settings search bar

3. Configure the following settings:
   - **Cline: OpenAI Base URL** - Set to your cluster URL (e.g., `http://your-cluster:8000/v1`)
   - **Cline: OpenAI API Key** - Set to your API key (or a dummy key like `na` if your cluster doesn't require one)
   - **Cline: OpenAI Model ID** - Set to your model name (e.g., `your-model-name`)

### 2. Edit settings.json Directly
Open `~/.config/Code/User/settings.json` and add:

```json
{
  "cline.openAiBaseUrl": "http://your-cluster-url:8000/v1",
  "cline.openAiApiKey": "your-api-key-here",
  "cline.openAiModelId": "your-model-name"
}
```

### 3. Project-level Configuration (.cline/settings.json)
Create a `.cline/settings.json` file in your project root:

```json
{
  "openAiBaseUrl": "http://your-cluster-url:8000/v1",
  "openAiApiKey": "your-api-key-here",
  "openAiModelId": "your-model-name"
}
```

## Requirements for Your LLM Endpoint

Your cluster's LLM must:
1. Be compatible with OpenAI's API format
2. Support the `/chat/completions` endpoint
3. Accept requests in this format:

```json
{
  "model": "your-model-name",
  "messages": [
    {"role": "system", "content": "..."},
    {"role": "user", "content": "..."}
  ]
}
```

## Common Endpoint Formats

| Platform | Base URL Format |
|----------|-----------------|
| Local Llama.cpp server | `http://localhost:8080/v1` |
| Text Generation WebUI | `http://localhost:5000/v1` |
| Ollama | `http://localhost:11434/v1` |
| Custom cluster | `http://your-cluster-ip:port/v1` |

## Troubleshooting

- **Connection refused**: Ensure your LLM server is running and accessible
- **401 Unauthorized**: Check your API key is correct
- **Model not found**: Verify the model name matches what's loaded on your cluster

## Testing Your Configuration

After setting up, you can test by:
1. Restarting VS Code
2. Opening Cline extension
3. Sending a test message to verify connectivity