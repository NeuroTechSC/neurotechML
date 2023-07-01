# 2022-23-NeuroTechX-Project (Software-Team)

## GUI

### How to start GUI with socket listener:
    python main.py

## Database

### How to start database
    docker compose up

## Known Issues:

- **CTRL+C** does not terminate socket listener on Windows.
    - Use `tasklist` to find the python PID and `taskkill /pid <PID> /f` to terminate the program