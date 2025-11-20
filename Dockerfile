FROM ollama/ollama:latest

# Set working directory
WORKDIR /root

# Pull models during build (optional, for faster startup)
RUN ollama pull embeddinggemma
RUN ollama pull gpt-oss:120b  # Your models

# Expose port
EXPOSE 11434

# Start Ollama server
CMD ["ollama", "serve"]
