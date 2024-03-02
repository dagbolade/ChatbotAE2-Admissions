# SolentBot: University Admissions Chatbot

SolentBot is an AI-driven chatbot designed to streamline university admissions and enrollment processes. It leverages LSTM neural networks for natural language processing and provides a Streamlit-based web interface for ease of use.

## Getting Started

Follow these instructions to set up and run SolentBot on your local machine for development and testing.

### Prerequisites

- Python 3.7 or later.

### Installation

1. Clone or download the repository to your local machine.
2. Navigate to the project directory.
3. (Optional) Create a virtual environment:
   ```sh
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # macOS/Linux
   source venv/bin/activate

4. Install the required dependencies:
   ```sh
   pip install -r requirements.txt


## Running SolentBot

To interact with SolentBot, you can use the console-based interface or the Streamlit web interface.

### Console Testing

  Execute `main.py` to start a console-based chat session:
    

### Streamlit web interface
Run chatbot_interface.py to launch the Streamlit web interface:

    ``` sh
    streamlit run chatbot_interface.py

Streamlit will automatically open the web interface in your default browser.
## Project Structure

Below is an outline of the key files and directories in the SolentBot project:

- `main.py`: The main Python script for console-based chatbot testing.
- `chatbot_interface.py`: The Streamlit application script for the web interface.
- `chatbot_model.ipynb`: Jupyter notebook detailing the chatbot model training.
- `new_train_file.ipynb`: Jupyter notebook for additional training procedures.
- `requirements.txt`: A list of dependencies

