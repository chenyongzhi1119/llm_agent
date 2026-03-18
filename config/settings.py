"""Project settings"""

# LM Studio local model config
LLM_BASE_URL = "http://localhost:1234/v1"
LLM_API_KEY = "lm-studio"  # LM Studio doesn't validate key, but OpenAI SDK requires non-empty
LLM_MODEL = "qwen2.5"  # Model name loaded in LM Studio, change as needed

# Agent config
MAX_REACT_STEPS = 10  # Max reasoning steps for ReAct loop
MAX_CONTEXT_MESSAGES = 20  # Max messages kept in short-term memory

# API config
API_HOST = "0.0.0.0"
API_PORT = 8000

# Security config
CODE_EXEC_TIMEOUT = 10  # Code execution timeout in seconds
ALLOWED_FILE_DIR = "./data"  # Restrict file read/write to this directory
