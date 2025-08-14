# SkillBridge - AI Talent Matching Assistant

## Overview

SkillBridge is an advanced AI-powered talent matching system that intelligently connects employees with ideal project roles based on skills, experience, and career aspirations. This comprehensive solution helps organizations optimize their workforce allocation while supporting employee growth and development.


## Key Features

### 🔍 Intelligent Matching Engine
- **Dual-direction matching**: Employee-to-role and role-to-employee matching
- **Multi-factor analysis**: Considers skills, certifications, experience, and career goals
- **Semantic understanding**: Goes beyond keyword matching using NLP techniques

### 📊 Workforce Analytics
- Real-time skill gap analysis
- Department-specific workforce insights
- Visual representation of organizational capabilities

### 👥 Employee-Centric Features
- Career goal alignment scoring
- Personalized role recommendations
- Development opportunity identification

### 🛠️ Management Tools
- Comprehensive employee and role management
- Customizable matching criteria
- Reporting and export capabilities

## Technology Stack

- **Core AI**: 
  - Scikit-learn for TF-IDF vectorization
  - Sentence Transformers for semantic analysis
  - Cosine similarity scoring

- **Frontend**:
  - PyQt5 for desktop interface
  - Modern UI components
  - Interactive data visualization

- **Backend**:
  - Python 3.8+
  - Dataclasses for data modeling
  - JSON for data serialization

## Installation

### Prerequisites
- Python 3.8 or later
- pip package manager

### Setup Instructions

1. Clone the repository:
   ```bash
   git clone https://github.com/your-repo/SkillBridge.git
   cd SkillBridge
   ```

2. Create and activate a virtual environment (recommended):
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows use `venv\Scripts\activate`
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Download the language model (first run will do this automatically):
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
1. Navigate to the Employees tab
2. Fill in employee details including:
   - Core skills and certifications
   - Career goals
   - Availability date
   - Preferences (location, department, etc.)
3. Click "Add Employee"

### Creating Roles
1. Go to the Roles tab
2. Define role requirements:
   - Required and nice-to-have skills
   - Mandatory certifications
   - Project timeline and location
3. Click "Add Role"

### Finding Matches
1. Select the Matching tab
2. Choose match direction:
   - "Employee → Roles" to find suitable positions for an employee
   - "Role → Employees" to find candidates for a role
3. Enter the ID and click "Find Matches"

### Analyzing Skill Gaps
1. Navigate to the Analysis tab
2. Optionally filter by department
3. Click "Analyze Skill Gaps" to view:
   - Skills in highest demand
   - Current capacity
   - Gap severity

## Sample Data

The application comes pre-loaded with sample data including:
- 3 employee profiles with diverse skill sets
- 3 project roles with different requirements
- Example matching results and gap analysis
---
**SkillBridge** - Bridging talent with opportunity through AI-powered matching.
