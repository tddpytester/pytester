# Copilot.api

the code is based on https://github.com/B00TK1D/copilot-api

Provides a simple HTTP API to interface with GitHub Copilot, including native GitHub authentication.

## Run Server
```
python3 api.py [port]
```

### Test
Send a POST request to `http://localhost:8080/api` with the following JSON body:
```json
{
    "prompt": "# Comment with a prompt\n\n",
    "language": "python"
}
```

## Usage
```
python query_apps.py
```

### Response
The response will be a plain text string containing the generated code.

In order to build a complete code snippet, iteratively append the generated code to the prompt and send it back to the API until the response is empty.
