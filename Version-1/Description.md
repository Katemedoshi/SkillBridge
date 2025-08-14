# SkillBridge - AI Talent Matching System

## Overview

SkillBridge is an intelligent AI-powered talent matching system that helps organizations optimally match employees with project roles based on skills, experience, certifications, and career goals. The system uses advanced NLP and machine learning techniques to provide data-driven recommendations for workforce allocation.


## Key Features

### 🎯 Intelligent Matching
- **Dual-direction matching**: 
  - Find best roles for an employee
  - Find best employees for a role
- **Multi-factor scoring**:
  - Skill matching (TF-IDF + cosine similarity)
  - Certification requirements
  - Career goal alignment (semantic similarity)
  - Location preferences
  - Availability constraints

### 📊 Analytics Dashboard
- Real-time skill gap analysis
- Department-specific insights
- Visual compatibility indicators

### 👥 Employee Management
- Comprehensive employee profiles
- Skill and certification tracking
- Career goal documentation
- Availability scheduling

### 📋 Role Management
- Detailed role requirements
- Skill priority classification
- Project timeline tracking
- Location specifications

## Technology Stack

### Core Matching Engine
- Python 3.8+
- Scikit-learn (TF-IDF vectorization)
- Sentence Transformers (semantic similarity)
- Cosine similarity scoring

### User Interface
- PyQt5 for desktop application
- Modern "Fusion" style
- Interactive tables and forms
- Color-coded results

### Data Management
- In-memory data structures
- JSON serialization
- Pandas for gap analysis

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

4. Run the application:
   ```bash
   python main.py
   ```

## Usage Guide

### Adding Employees
1. Navigate to the Employees tab
2. Fill in employee details including:
   - Basic information (ID, name)
   - Skills and certifications
   - Career goals
   - Availability date
   - Preferences (location, department)
3. Click "Add Employee"

### Creating Roles
1. Go to the Roles tab
2. Define role requirements:
   - Required and nice-to-have skills
   - Mandatory certifications
   - Project timeline (start date, duration)
   - Location and priority
3. Click "Add Role"

### Finding Matches
1. Select the Matching tab
2. Choose match direction:
   - "Employee to Roles" to find suitable positions
   - "Role to Employees" to find candidates
3. Enter the ID and click "Find Matches"
4. View color-coded results

### Analyzing Skill Gaps
1. Navigate to the Analysis tab
2. Optionally filter by department
3. Click "Analyze Skill Gaps"
4. Review the detailed gap report

## Sample Data

The system comes pre-loaded with sample data including:

- 2 employee profiles:
  - ML Engineer with Python/AWS skills
  - Cloud Architect with Java/AWS expertise

- 2 project roles:
  - ML Engineer position
  - Cloud Architect role



**SkillBridge** - Bridging talent with opportunity through intelligent matching.
