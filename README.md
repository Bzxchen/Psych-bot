# PsychBot - AI Psychology Assistant

A comforting, informative AI assistant that answers questions based on psychology textbooks and resources. The bot is designed to communicate in a warm, empathetic, psychologist-like tone while providing accurate, evidence-based information from multiple psychology sources.

## Features

- 📚 **PDF Textbook Ingestion**: Extract and process text from psychology textbook PDFs
- 💬 **Interactive Q&A**: Ask questions and receive thoughtful, comforting responses
- 🧠 **OpenAI Integration**: Powered by GPT-4 (or GPT-3.5-turbo) for intelligent responses
- 💙 **Empathetic Tone**: Designed to be warm, supportive, and non-judgmental

## Setup

### 1. Create a Virtual Environment

Open your terminal (WSL) in the project directory and run:

```bash
python3 -m venv venv
```

This creates a `venv` folder that will contain your isolated Python environment.

### 2. Activate the Virtual Environment

```bash
source venv/bin/activate
```

You'll know it's activated when you see `(venv)` at the start of your terminal prompt.

### 3. Install Dependencies

With the virtual environment activated, run:

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### 4. Set Up OpenAI API Key

Create a `.env` file in the project root directory with the following content:

```
OPENAI_API_KEY=your_api_key_here
```

Replace `your_api_key_here` with your actual OpenAI API key. You can get your API key from: https://platform.openai.com/api-keys

Alternatively, you can set the `OPENAI_API_KEY` environment variable directly in your shell.

### 5. Run the Bot

Make sure your virtual environment is activated (you should see `(venv)` in your prompt), then run:

```bash
python3 psych_bot.py
```

### Activating the Virtual Environment (For Future Sessions)

Each time you want to use the bot, you'll need to activate the virtual environment first:

```bash
source venv/bin/activate
```

Then run `python3 psych_bot.py` to start the bot.

## Usage

1. Run the bot - it will automatically load all PDFs from the data folder
2. Start asking questions about psychology!
3. Type 'quit', 'exit', or 'bye' to end the conversation

### Example Questions

**Conceptual:**
- "What is cognitive behavioral therapy?"
- "How does memory work?"
- "What is the difference between anxiety and panic disorders?"

**Applied:**
- "Can you give me an example of how CBT is used in practice?"
- "What are some coping strategies for anxiety?"
- "How would a therapist apply exposure therapy?"

**Emotional:**
- "I've been feeling really anxious lately"
- "I'm struggling with depression and don't know what to do"
- "How can I cope with overwhelming stress?"

## Project Structure

```
Psych-bot/
├── psych_bot.py              # Main bot implementation
├── question_classifier.py    # Question type classification and response structuring
├── training_data_template.json  # Template for future ML training data
├── requirements.txt          # Python dependencies
├── README.md                 # This file
├── .env                      # Your API keys (create this)
├── data/                     # Data folder (create this)
│   ├── DSM-5-TR.pdf         # Psychology resources (place PDFs here)
│   └── ...                   # Add more PDF resources here
└── venv/                     # Virtual environment (create with: python3 -m venv venv)
```

## Features

### Question Classification & Tailored Responses

The bot automatically classifies questions into different types and structures responses accordingly:

- **Conceptual Questions**: Definitions, theories - receives clear, textbook-grounded explanations
- **Applied Questions**: Real-life examples - receives practical applications and scenarios
- **Emotional Questions**: Personal distress - receives extra warmth, validation, and support
- **Clarification Questions**: Follow-ups - receives thorough, detailed explanations

### Safety & Professional Resources

**Important**: This bot is designed for educational purposes and is NOT a replacement for professional mental health care.

- **Crisis Detection**: The bot automatically detects crisis situations (suicidal thoughts, self-harm, etc.) and immediately provides emergency resources
- **Professional Redirects**: For emotional and applied questions, the bot includes information about professional resources
- **Safety Disclaimers**: Appropriate disclaimers are included when discussing mental health topics

### Technical Features

- **Vector Embeddings & Semantic Search**: The bot uses OpenAI embeddings to create a searchable index of all psychology resources. It automatically finds the most relevant sections for each question, avoiding token limits and improving accuracy.
- **Caching**: Embeddings are cached after the first run, so subsequent startups are much faster. Cache files are named `embeddings_cache_*.pkl`.
- **Question Classifier**: Currently uses rule-based classification. Structure is in place for future ML model training (Logistic Regression, SVM, or fine-tuned BERT).

## Notes

- The default model is GPT-4. You can change it to `gpt-3.5-turbo` in the code for cost savings.
- The bot is designed to be comforting and supportive while maintaining professional accuracy.
- The first time you run the bot with PDFs, it will take some time to create embeddings. This is a one-time process per PDF file.
- **Future ML Training**: The `training_data_template.json` file provides a structure for collecting training data to replace the rule-based classifier with an ML model.

## Work in Progress / Future Features

- **ML Model Training**: Currently working on training an ML model for question classification to replace the rule-based system. Collecting training data and preparing for model integration (Logistic Regression, SVM, or fine-tuned BERT).
- **Expanding Resources**: Continuously adding more psychology textbooks and resources to the `data/` folder to expand the knowledge base.
- **Frontend Development**: Planning to develop a web-based frontend interface for easier access and improved user experience.
- **Weaviate Integration**: Planning to integrate Weaviate vector database for more efficient and scalable embedding storage and retrieval, replacing the current pickle-based caching system.

## Requirements

- Python 3.7+
- OpenAI API key
- PDF resources (place PDF files in the `data/` folder)

## License

This is a personal project for educational purposes.
