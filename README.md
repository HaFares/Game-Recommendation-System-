# 🎮 Steam Games Analytics & Recommendation System

A comprehensive web application built with Streamlit that provides interactive analytics, AI-powered game recommendations, and an intelligent chatbot for exploring Steam's vast gaming library.

![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)
![Streamlit](https://img.shields.io/badge/Streamlit-1.45.1-FF4B4B.svg)
![License](https://img.shields.io/badge/License-MIT-green.svg)

## ✨ Features

### 📊 Interactive Dashboard
- **Comprehensive Analytics**: Visualize 15,000+ Steam games with interactive charts
- **Price Analysis**: Distribution of game prices and discount patterns
- **Genre Insights**: Top 15 game genres with detailed breakdowns
- **Platform Support**: Cross-platform availability statistics (Windows, macOS, Linux, SteamOS)
- **Release Trends**: Monthly and yearly game release patterns
- **Developer Statistics**: Top developers and publishers by game count
- **Review Analysis**: User review distribution and sentiment
- **Word Cloud**: Visual representation of game descriptions and themes

### 🎯 AI-Powered Recommendation System
- **Smart Recommendations**: Content-based filtering using cosine similarity
- **Personalized Results**: Get 1-15 similar game suggestions
- **Rich Game Cards**: Beautiful UI with game images, pricing, and discounts
- **Detailed Information**: Full game details including:
  - Release dates
  - Developer and publisher info
  - Original pricing and current discounts
  - Direct Steam store links
- **Responsive Design**: Mobile-friendly interface with modern aesthetics

### 🤖 Intelligent Game Chatbot
- **Conversational AI**: Powered by Llama 3 (8B parameters) via Groq
- **RAG Architecture**: Retrieval-Augmented Generation using FAISS vector database
- **Natural Language Understanding**: Ask questions in plain English
- **Game Discovery**: Get personalized recommendations through conversation
- **Context-Aware**: Remembers chat history for coherent dialogues
- **Rich Responses**: Detailed game information and suggestions

## 🚀 Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager
- Git

### Setup Instructions

1. **Clone the repository**
```bash
git clone https://github.com/yourusername/steam-games-analytics.git
cd steam-games-analytics
```

2. **Create a virtual environment**
```bash
python -m venv venv

# On Windows
venv\Scripts\activate

# On macOS/Linux
source venv/bin/activate
```

3. **Install dependencies**
```bash
pip install -r requirements.txt
```

4. **Set up API keys**

Create a `.env` file in the root directory or set environment variables:
```bash
export HUGGINGFACEHUB_API_TOKEN="your_huggingface_token"
export GROQ_API_KEY="your_groq_api_key"
```

5. **Prepare data files**

Ensure the following files are in your project directory:
- `game_data_for_dashboard.csv` - Main dataset for analytics
- `games_data_recomendation.csv` - Dataset for recommendations
- `similarity_finale.pkl` - Pre-computed similarity matrix
- `faiss_index/` - FAISS vector database directory

## 📁 Project Structure

```
steam-games-analytics/
│
├── app.py                          # Main Streamlit application
├── requirements.txt                # Python dependencies
├── README.md                       # Project documentation
│
├── data/
│   ├── game_data_for_dashboard.csv
│   ├── games_data_recomendation.csv
│   └── similarity_finale.pkl
│
├── faiss_index/                    # Vector database for chatbot
│   ├── index.faiss
│   └── index.pkl
│
└── assets/                         # Images and static files
```

## 🎯 Usage

### Running the Application

```bash
streamlit run app.py
```

The application will open in your default browser at `http://localhost:8501`

### Navigation

Use the sidebar to switch between three main sections:

1. **Dashboard** - Explore comprehensive game analytics
2. **Recommendation System** - Find similar games based on your preferences
3. **Chatbot** - Chat with AI to discover new games

## 🛠️ Technologies Used

### Core Framework
- **Streamlit** - Web application framework
- **Python 3.8+** - Programming language

### Data Processing & Visualization
- **Pandas** - Data manipulation and analysis
- **NumPy** - Numerical computing
- **Plotly** - Interactive visualizations
- **Matplotlib & Seaborn** - Statistical graphics
- **WordCloud** - Text visualization

### Machine Learning & AI
- **LangChain** - LLM application framework
- **Groq** - Fast LLM inference (Llama 3)
- **HuggingFace** - Embeddings (all-MiniLM-L6-v2)
- **FAISS** - Vector similarity search
- **Sentence Transformers** - Text embeddings

### UI Components
- **streamlit-shadcn-ui** - Enhanced UI components

## 📊 Data Sources

The application uses Steam game data including:
- Game titles and descriptions
- Pricing information and discounts
- Release dates
- Developer and publisher details
- Platform compatibility
- User reviews and ratings
- Genre classifications
- Game images and Steam store links

## 🤝 Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Steam for providing the gaming data
- Groq for fast LLM inference
- HuggingFace for embedding models
- The Streamlit team for the amazing framework
- The open-source community for various libraries used

## 📧 Contact

Your Name - [@yourtwitter](https://twitter.com/yourtwitter)

Project Link: [https://github.com/yourusername/steam-games-analytics](https://github.com/yourusername/steam-games-analytics)

## 🔮 Future Enhancements

- [ ] User authentication and personalized profiles
- [ ] Save favorite games and recommendations
- [ ] Advanced filtering options (by genre, price range, platform)
- [ ] Integration with Steam API for real-time data
- [ ] Collaborative filtering recommendations
- [ ] Game comparison features
- [ ] Export analytics reports
- [ ] Multi-language support

## 📸 Screenshots

### Dashboard
![Dashboard](assets/dashboard.png)

### Recommendation System
![Recommendations](assets/recommendations.png)

### AI Chatbot
![Chatbot](assets/chatbot.png)

---

⭐ If you find this project useful, please consider giving it a star!

Made with ❤️ and ☕
