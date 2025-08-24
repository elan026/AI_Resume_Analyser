📄 AI Resume AnalyserAn AI-powered Resume Analyser built with LangChain and Streamlit, designed to help job seekers and recruiters evaluate resumes against job descriptions. The app extracts key skills, experiences, and qualifications from resumes, compares them with job requirements, and generates insights, strengths, weaknesses, and improvement suggestions.Supports both online (OpenAI) and offline (Hugging Face / Instructor) embeddings.🚀 Features✅ Upload Resumes: Supports PDF, DOCX, and TXT formats.✅ Content Analysis: Extracts and analyzes resume content using LangChain.✅ Job Description Comparison: Matches resume details against a provided job description.✅ ATS Evaluation: Provides a match percentage and an ATS-style evaluation.✅ Improvement Tips: Generates actionable suggestions for candidates.✅ Flexible Embeddings: Option to use the OpenAI API or free, offline Hugging Face embeddings.✅ Modern UI: A smooth and interactive user interface built with Streamlit.🛠️ Tech StackPython: 3.11+ (recommended for 2025 compatibility)LLM Orchestration: LangChainWeb UI: StreamlitEmbeddings: Hugging Face TransformersText Extraction: PyPDF2 / python-docxVector Search: FAISSEnvironment: python-dotenv📂 Project Structureai-resume-analyser/
│
├── app.py                # Streamlit frontend
├── embeddings.py         # Embeddings setup (OpenAI / HuggingFace)
├── resume_parser.py      # Extract text from resumes
├── analyser.py           # Core LangChain logic
├── requirements.txt      # Python dependencies
├── README.md             # Documentation
├── .env.example          # Sample environment variables
│
└── data/
    ├── sample_resume.pdf
    └── job_description.txt
⚙️ InstallationClone the repository:git clone https://github.com/yourusername/ai-resume-analyser.git
cd ai-resume-analyser
Create and activate a virtual environment:# Mac/Linux
python -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate
Install dependencies:pip install -r requirements.txt
🔑 Environment SetupCreate a .env file in the project root and add your API keys.# For OpenAI (optional, paid)
OPENAI_API_KEY=your_openai_key_here

# For Hugging Face (free/offline mode)
HF_API_KEY=your_huggingface_key_here
If you only want to use the offline mode, you can use Hugging Face’s free embeddings (e.g., sentence-transformers/all-MiniLM-L6-v2) without an API key.📝 Configuring EmbeddingsYou can easily switch between embedding providers in embeddings.py:from langchain.embeddings import HuggingFaceEmbeddings
# from langchain_openai import OpenAIEmbeddings  # Optional if using OpenAI

def get_embeddings(provider="huggingface"):
    if provider == "openai":
        return OpenAIEmbeddings(model="text-embedding-3-large")
    else:
        return HuggingFaceEmbeddings(
            model_name="sentence-transformers/all-MiniLM-L6-v2"
        )
▶️ Run the AppLaunch the Streamlit application with the following command:streamlit run app.py
Navigate to http://localhost:8501 in your web browser.🎯 UsageUpload your resume (PDF/DOCX/TXT).Paste or upload a job description.The analyser will:Extract resume details.Compare them against the job requirements.Show a match percentage and identify missing skills.Provide ATS-style suggestions for improvement.📊 Example OutputMatch Score: 78%

Key Strengths: Python, Machine Learning, Problem-Solving

Missing Skills: Docker, AWS

Suggestions: Consider adding specific examples of cloud deployment experience to better align with the job description.
📦 RequirementsSee the requirements.txt file for a full list of dependencies.streamlitlangchainhuggingface-hubtransformerssentence-transformersfaiss-cpupython-dotenvPyPDF2python-docx🤝 ContributingContributions are welcome! Feel free to fork the repository, make your changes, and submit a pull request.📜 LicenseThis project is licensed under the MIT License. © 2025
