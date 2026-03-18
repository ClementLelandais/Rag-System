import uvicorn
from dotenv import load_dotenv
import argparse
import os

load_dotenv()

DEFAULT_HOST  = os.getenv("HOST", "0.0.0.0") 
DEFAULT_PORT = int(os.getenv("PORT", "8000"))

def main():
    parser = argparse.ArgumentParser(description="Run backend of RAG System")
    parser.add_argument("--host", type=str, default=DEFAULT_HOST, help="Host to run the server on")
    parser.add_argument("--port", type=int, default=DEFAULT_PORT, help="Port to run the server on")
    parser.add_argument("--dev", action="store_true", help="Run in development mode")
    args = parser.parse_args()

    # Lance le serveur Uvicorn avec les paramètres spécifiés
    # dans les arguments + environnement 
    uvicorn.run(
        "backend.main:app",
        host=args.host,
        port=args.port,
        reload=args.dev,
    )

if __name__ == "__main__":
    main()