# SkillBridge - AI Talent Matching System

## Overview

SkillBridge is an advanced AI-powered talent matching platform that intelligently connects employees with ideal project roles based on skills, experience, certifications, and career aspirations. The system uses sophisticated NLP and machine learning techniques to optimize workforce allocation and career development.


## Key Features

### 🧠 Intelligent Matching Engine
- **Bi-directional matching**:
  - Employee-to-Role recommendations
  - Role-to-Employee candidate search
- **Multi-dimensional scoring**:
  - Skill matching (TF-IDF + cosine similarity)
  - Certification requirements
  - Career goal alignment (semantic similarity)
  - Location preferences
  - Availability constraints

### 📊 Advanced Analytics
- Real-time organizational skill gap analysis
- Department-specific workforce insights
- Visual data representations
- Priority-ranked gap reporting

### 👥 Employee Management
- Comprehensive profile system
- Skill and certification tracking
- Career goal documentation
- Availability scheduling
- Preference management

### 📋 Role Management
- Detailed role requirements
- Required vs. nice-to-have skills
- Project timeline tracking
- Location and priority settings

## Technology Stack

### Core AI Components
- **Natural Language Processing**:
  - Scikit-learn TF-IDF vectorization
  - Sentence Transformers (all-MiniLM-L6-v2)
  - Cosine similarity scoring

### User Interface
- **PyQt5** for cross-platform desktop application
- Modern "Fusion" styling
- Custom interactive widgets
- Color-coded visualizations

### Data Processing
- Python dataclasses for data modeling
- JSON serialization
- Pandas for gap analysis
- Matplotlib for visualizations

## Installation

### Prerequisites
- Python 3.8+
- pip package manager

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/SkillBridge.git
   cd SkillBridge
   ```

2. Create and activate a virtual environment:
   ```bash
   python -m venv venv
   source venv/bin/activate  # Windows: venv\Scripts\activate
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. (Optional) Download the language model:
   ```python
   from sentence_transformers import SentenceTransformer
   SentenceTransformer('all-MiniLM-L6-v2')
   ```

5. Run the application:
   ```bash
   python main.py
   ```

## Usage Guide

### Adding Employees
1. Navigate to the "👥 Employees" tab
2. Fill in employee details:
   - Required: ID, Name
   - Skills (comma-separated)
   - Experience (years)
   - Certifications
   - Career goals
   - Availability date (YYYY-MM-DD)
   - Preferences (JSON format)
3. Click "Add Employee"

### Creating Roles
1. Go to the "📋 Roles" tab
2. Enter role information:
   - Required: Role ID, Title
   - Project ID
   - Required skills
   - Nice-to-have skills
   - Required certifications
   - Start date and duration
   - Location and priority
3. Click "Add Role"

### Finding Matches
1. Select the "✨ Matching" tab
2. Choose match direction:
   - "Employee → Roles" to find positions for an employee
   - "Role → Employees" to find candidates for a role
3. Enter the ID and click "Find Matches"
4. View color-coded results with compatibility scores

### Analyzing Skill Gaps
1. Navigate to the "📈 Analysis" tab
2. Optionally enter a department filter
3. Click "Analyze Gaps"
4. Review the detailed report showing:
   - Skills in demand
   - Current capacity
   - Gap severity
   - Priority indicators

## Sample Data

The system includes sample data for demonstration:

**Employees:**
- John Doe: ML Engineer with Python/AWS skills
- Jane Smith: Cloud Architect with Java expertise
- Mike Johnson: Full-stack Developer with React/Node skills

**Roles:**
- Senior ML Engineer position
- Cloud Solutions Architect role
- Full-Stack Developer position


**SkillBridge** - Revolutionizing talent allocation through AI-powered matching and workforce analytics.
