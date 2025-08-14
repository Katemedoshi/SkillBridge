import sys
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datetime import datetime
from typing import List, Dict, Optional
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, 
                             QLabel, QLineEdit, QPushButton, QTextEdit, QListWidget, 
                             QTabWidget, QComboBox, QTableWidget, QTableWidgetItem, 
                             QHeaderView, QMessageBox)
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont

class Employee:
    def __init__(self, employee_id: str, name: str, skills: List[str], 
                 experience_years: int, certifications: List[str], 
                 career_goals: List[str], past_projects: List[str],
                 availability_date: str, preferences: Dict):
        self.employee_id = employee_id
        self.name = name
        self.skills = skills
        self.experience_years = experience_years
        self.certifications = certifications
        self.career_goals = career_goals
        self.past_projects = past_projects
        self.availability_date = datetime.strptime(availability_date, "%Y-%m-%d")
        self.preferences = preferences
        
    def to_dict(self):
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "skills": self.skills,
            "experience_years": self.experience_years,
            "certifications": self.certifications,
            "career_goals": self.career_goals,
            "past_projects": self.past_projects,
            "availability_date": self.availability_date.strftime("%Y-%m-%d"),
            "preferences": self.preferences
        }

class ProjectRole:
    def __init__(self, role_id: str, project_id: str, title: str, 
                 required_skills: List[str], nice_to_have_skills: List[str],
                 required_certifications: List[str], start_date: str,
                 duration_weeks: int, location: str, priority: int):
        self.role_id = role_id
        self.project_id = project_id
        self.title = title
        self.required_skills = required_skills
        self.nice_to_have_skills = nice_to_have_skills
        self.required_certifications = required_certifications
        self.start_date = datetime.strptime(start_date, "%Y-%m-%d")
        self.duration_weeks = duration_weeks
        self.location = location
        self.priority = priority
        
    def to_dict(self):
        return {
            "role_id": self.role_id,
            "project_id": self.project_id,
            "title": self.title,
            "required_skills": self.required_skills,
            "nice_to_have_skills": self.nice_to_have_skills,
            "required_certifications": self.required_certifications,
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "duration_weeks": self.duration_weeks,
            "location": self.location,
            "priority": self.priority
        }

class AIMatchingEngine:
    def __init__(self):
        # Initialize models and components
        self.skill_vectorizer = TfidfVectorizer()
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.employees = []
        self.roles = []
        self.skill_matrix = None
        self.skill_vocab = None
        
    def add_employee(self, employee: Employee):
        """Add an employee to the matching pool"""
        self.employees.append(employee)
        
    def add_role(self, role: ProjectRole):
        """Add a project role to the matching pool"""
        self.roles.append(role)
        
    def build_skill_matrix(self):
        """Build TF-IDF matrix for all skills"""
        all_skills = []
        for emp in self.employees:
            all_skills.append(" ".join(emp.skills))
        for role in self.roles:
            all_skills.append(" ".join(role.required_skills + role.nice_to_have_skills))
            
        self.skill_matrix = self.skill_vectorizer.fit_transform(all_skills)
        self.skill_vocab = self.skill_vectorizer.vocabulary_
        
    def calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between two texts using SBERT"""
        embeddings = self.sbert_model.encode([text1, text2])
        return cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
    
    def match_employee_to_roles(self, employee_id: str, top_n: int = 5) -> List[Dict]:
        """Find best matching roles for a specific employee"""
        employee = next((emp for emp in self.employees if emp.employee_id == employee_id), None)
        if not employee:
            return []
            
        results = []
        for role in self.roles:
            # Skip roles that start before employee is available
            if role.start_date < employee.availability_date:
                continue
                
            # Calculate skill match score
            emp_skills_vec = self.skill_vectorizer.transform([" ".join(employee.skills)])
            role_skills_vec = self.skill_vectorizer.transform([" ".join(role.required_skills)])
            skill_score = cosine_similarity(emp_skills_vec, role_skills_vec)[0][0]
            
            # Calculate certification match
            cert_match = len(set(role.required_certifications) & set(employee.certifications)) / \
                        max(1, len(role.required_certifications))
                        
            # Calculate career goal alignment (semantic similarity)
            career_alignment = 0
            if employee.career_goals:
                career_alignment = self.calculate_semantic_similarity(
                    " ".join(employee.career_goals),
                    role.title + " " + " ".join(role.required_skills)
                )
                
            # Location preference match
            location_match = 1 if not employee.preferences.get('location') or \
                                employee.preferences['location'] == role.location else 0
                                
            # Calculate overall score (weighted average)
            overall_score = (
                0.5 * skill_score + 
                0.2 * cert_match + 
                0.2 * career_alignment + 
                0.1 * location_match
            )
            
            results.append({
                "role": role.to_dict(),
                "score": overall_score,
                "skill_match": skill_score,
                "cert_match": cert_match,
                "career_alignment": career_alignment,
                "location_match": location_match,
                "match_details": f"Skills: {skill_score:.2f}, Certs: {cert_match:.2f}, Career: {career_alignment:.2f}, Location: {location_match}"
            })
        
        # Sort by score and return top N
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]
    
    def match_role_to_employees(self, role_id: str, top_n: int = 5) -> List[Dict]:
        """Find best matching employees for a specific role"""
        role = next((r for r in self.roles if r.role_id == role_id), None)
        if not role:
            return []
            
        results = []
        for employee in self.employees:
            # Skip employees not available by role start date
            if role.start_date < employee.availability_date:
                continue
                
            # Calculate skill match score
            emp_skills_vec = self.skill_vectorizer.transform([" ".join(employee.skills)])
            role_skills_vec = self.skill_vectorizer.transform([" ".join(role.required_skills)])
            skill_score = cosine_similarity(emp_skills_vec, role_skills_vec)[0][0]
            
            # Calculate certification match
            cert_match = len(set(role.required_certifications) & set(employee.certifications)) / \
                        max(1, len(role.required_certifications))
                        
            # Calculate career goal alignment (semantic similarity)
            career_alignment = 0
            if employee.career_goals:
                career_alignment = self.calculate_semantic_similarity(
                    " ".join(employee.career_goals),
                    role.title + " " + " ".join(role.required_skills)
                )
                
            # Location preference match
            location_match = 1 if not employee.preferences.get('location') or \
                                employee.preferences['location'] == role.location else 0
                                
            # Calculate overall score (weighted average)
            overall_score = (
                0.5 * skill_score + 
                0.2 * cert_match + 
                0.2 * career_alignment + 
                0.1 * location_match
            )
            
            results.append({
                "employee": employee.to_dict(),
                "score": overall_score,
                "skill_match": skill_score,
                "cert_match": cert_match,
                "career_alignment": career_alignment,
                "location_match": location_match,
                "match_details": f"Skills: {skill_score:.2f}, Certs: {cert_match:.2f}, Career: {career_alignment:.2f}, Location: {location_match}"
            })
        
        # Sort by score and return top N
        results.sort(key=lambda x: x['score'], reverse=True)
        return results[:top_n]
    
    def analyze_skill_gaps(self, department: str = None) -> Dict:
        """Analyze skill gaps across the organization"""
        required_skills = []
        employee_skills = []
        
        for role in self.roles:
            required_skills.extend(role.required_skills)
            
        for emp in self.employees:
            if department and emp.preferences.get('department') != department:
                continue
            employee_skills.extend(emp.skills)
            
        # Count skill occurrences
        required_counts = pd.Series(required_skills).value_counts().to_dict()
        employee_counts = pd.Series(employee_skills).value_counts().to_dict()
        
        # Calculate gaps
        skill_gaps = {}
        for skill, count in required_counts.items():
            available = employee_counts.get(skill, 0)
            gap = max(0, count - available)
            if gap > 0:
                skill_gaps[skill] = {
                    "required": count,
                    "available": available,
                    "gap": gap
                }
                
        return skill_gaps

class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.engine = AIMatchingEngine()
        self.init_ui()
        self.load_sample_data()
        
    def init_ui(self):
        self.setWindowTitle("AI Agentic Matching System")
        self.setGeometry(100, 100, 1200, 800)
        
        # Central widget and layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        
        # Create tabs
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs)
        
        # Add tabs
        self.create_employee_tab()
        self.create_role_tab()
        self.create_matching_tab()
        self.create_analysis_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_employee_tab(self):
        """Create the employee management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Form for adding employees
        form_layout = QVBoxLayout()
        
        # Employee ID
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Employee ID:"))
        self.emp_id_input = QLineEdit()
        id_layout.addWidget(self.emp_id_input)
        form_layout.addLayout(id_layout)
        
        # Name
        name_layout = QHBoxLayout()
        name_layout.addWidget(QLabel("Name:"))
        self.emp_name_input = QLineEdit()
        name_layout.addWidget(self.emp_name_input)
        form_layout.addLayout(name_layout)
        
        # Skills
        skills_layout = QHBoxLayout()
        skills_layout.addWidget(QLabel("Skills (comma separated):"))
        self.emp_skills_input = QLineEdit()
        skills_layout.addWidget(self.emp_skills_input)
        form_layout.addLayout(skills_layout)
        
        # Experience
        exp_layout = QHBoxLayout()
        exp_layout.addWidget(QLabel("Experience (years):"))
        self.emp_exp_input = QLineEdit()
        exp_layout.addWidget(self.emp_exp_input)
        form_layout.addLayout(exp_layout)
        
        # Certifications
        cert_layout = QHBoxLayout()
        cert_layout.addWidget(QLabel("Certifications (comma separated):"))
        self.emp_cert_input = QLineEdit()
        cert_layout.addWidget(self.emp_cert_input)
        form_layout.addLayout(cert_layout)
        
        # Career Goals
        goals_layout = QHBoxLayout()
        goals_layout.addWidget(QLabel("Career Goals (comma separated):"))
        self.emp_goals_input = QLineEdit()
        goals_layout.addWidget(self.emp_goals_input)
        form_layout.addLayout(goals_layout)
        
        # Availability
        avail_layout = QHBoxLayout()
        avail_layout.addWidget(QLabel("Availability Date (YYYY-MM-DD):"))
        self.emp_avail_input = QLineEdit()
        avail_layout.addWidget(self.emp_avail_input)
        form_layout.addLayout(avail_layout)
        
        # Preferences
        pref_layout = QVBoxLayout()
        pref_layout.addWidget(QLabel("Preferences (JSON format):"))
        self.emp_pref_input = QTextEdit()
        self.emp_pref_input.setPlaceholderText('{"location": "New York", "department": "AI"}')
        pref_layout.addWidget(self.emp_pref_input)
        form_layout.addLayout(pref_layout)
        
        # Add button
        add_btn = QPushButton("Add Employee")
        add_btn.clicked.connect(self.add_employee)
        form_layout.addWidget(add_btn)
        
        # Employee list
        self.employee_list = QTableWidget()
        self.employee_list.setColumnCount(4)
        self.employee_list.setHorizontalHeaderLabels(["ID", "Name", "Skills", "Experience"])
        self.employee_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.employee_list.setSelectionBehavior(QTableWidget.SelectRows)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_employee_list)
        
        layout.addLayout(form_layout)
        layout.addWidget(refresh_btn)
        layout.addWidget(self.employee_list)
        
        self.tabs.addTab(tab, "Employees")
        
    def create_role_tab(self):
        """Create the role management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Form for adding roles
        form_layout = QVBoxLayout()
        
        # Role ID
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("Role ID:"))
        self.role_id_input = QLineEdit()
        id_layout.addWidget(self.role_id_input)
        form_layout.addLayout(id_layout)
        
        # Project ID
        proj_layout = QHBoxLayout()
        proj_layout.addWidget(QLabel("Project ID:"))
        self.role_proj_input = QLineEdit()
        proj_layout.addWidget(self.role_proj_input)
        form_layout.addLayout(proj_layout)
        
        # Title
        title_layout = QHBoxLayout()
        title_layout.addWidget(QLabel("Title:"))
        self.role_title_input = QLineEdit()
        title_layout.addWidget(self.role_title_input)
        form_layout.addLayout(title_layout)
        
        # Required Skills
        req_skills_layout = QHBoxLayout()
        req_skills_layout.addWidget(QLabel("Required Skills (comma separated):"))
        self.role_req_skills_input = QLineEdit()
        req_skills_layout.addWidget(self.role_req_skills_input)
        form_layout.addLayout(req_skills_layout)
        
        # Nice-to-have Skills
        nice_skills_layout = QHBoxLayout()
        nice_skills_layout.addWidget(QLabel("Nice-to-have Skills (comma separated):"))
        self.role_nice_skills_input = QLineEdit()
        nice_skills_layout.addWidget(self.role_nice_skills_input)
        form_layout.addLayout(nice_skills_layout)
        
        # Required Certifications
        cert_layout = QHBoxLayout()
        cert_layout.addWidget(QLabel("Required Certifications (comma separated):"))
        self.role_cert_input = QLineEdit()
        cert_layout.addWidget(self.role_cert_input)
        form_layout.addLayout(cert_layout)
        
        # Start Date
        start_layout = QHBoxLayout()
        start_layout.addWidget(QLabel("Start Date (YYYY-MM-DD):"))
        self.role_start_input = QLineEdit()
        start_layout.addWidget(self.role_start_input)
        form_layout.addLayout(start_layout)
        
        # Duration
        dur_layout = QHBoxLayout()
        dur_layout.addWidget(QLabel("Duration (weeks):"))
        self.role_dur_input = QLineEdit()
        dur_layout.addWidget(self.role_dur_input)
        form_layout.addLayout(dur_layout)
        
        # Location
        loc_layout = QHBoxLayout()
        loc_layout.addWidget(QLabel("Location:"))
        self.role_loc_input = QLineEdit()
        loc_layout.addWidget(self.role_loc_input)
        form_layout.addLayout(loc_layout)
        
        # Priority
        pri_layout = QHBoxLayout()
        pri_layout.addWidget(QLabel("Priority (1-5):"))
        self.role_pri_input = QLineEdit()
        pri_layout.addWidget(self.role_pri_input)
        form_layout.addLayout(pri_layout)
        
        # Add button
        add_btn = QPushButton("Add Role")
        add_btn.clicked.connect(self.add_role)
        form_layout.addWidget(add_btn)
        
        # Role list
        self.role_list = QTableWidget()
        self.role_list.setColumnCount(4)
        self.role_list.setHorizontalHeaderLabels(["ID", "Title", "Required Skills", "Location"])
        self.role_list.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.role_list.setSelectionBehavior(QTableWidget.SelectRows)
        
        # Refresh button
        refresh_btn = QPushButton("Refresh List")
        refresh_btn.clicked.connect(self.refresh_role_list)
        
        layout.addLayout(form_layout)
        layout.addWidget(refresh_btn)
        layout.addWidget(self.role_list)
        
        self.tabs.addTab(tab, "Roles")
        
    def create_matching_tab(self):
        """Create the matching tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Matching type selection
        match_type_layout = QHBoxLayout()
        match_type_layout.addWidget(QLabel("Match Type:"))
        self.match_type_combo = QComboBox()
        self.match_type_combo.addItems(["Employee to Roles", "Role to Employees"])
        match_type_layout.addWidget(self.match_type_combo)
        
        # ID input
        id_layout = QHBoxLayout()
        id_layout.addWidget(QLabel("ID to Match:"))
        self.match_id_input = QLineEdit()
        id_layout.addWidget(self.match_id_input)
        
        # Match button
        match_btn = QPushButton("Find Matches")
        match_btn.clicked.connect(self.find_matches)
        
        # Results table
        self.match_results = QTableWidget()
        self.match_results.setColumnCount(4)
        self.match_results.setHorizontalHeaderLabels(["Match", "Score", "Details", "Compatibility"])
        self.match_results.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        # Build matrix button
        build_btn = QPushButton("Build Skill Matrix")
        build_btn.clicked.connect(self.build_matrix)
        
        layout.addLayout(match_type_layout)
        layout.addLayout(id_layout)
        layout.addWidget(match_btn)
        layout.addWidget(build_btn)
        layout.addWidget(self.match_results)
        
        self.tabs.addTab(tab, "Matching")
        
    def create_analysis_tab(self):
        """Create the analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Department filter
        dept_layout = QHBoxLayout()
        dept_layout.addWidget(QLabel("Department Filter:"))
        self.dept_filter_input = QLineEdit()
        self.dept_filter_input.setPlaceholderText("Leave blank for all departments")
        dept_layout.addWidget(self.dept_filter_input)
        
        # Analyze button
        analyze_btn = QPushButton("Analyze Skill Gaps")
        analyze_btn.clicked.connect(self.analyze_gaps)
        
        # Results display
        self.gap_analysis_results = QTextEdit()
        self.gap_analysis_results.setReadOnly(True)
        
        layout.addLayout(dept_layout)
        layout.addWidget(analyze_btn)
        layout.addWidget(self.gap_analysis_results)
        
        self.tabs.addTab(tab, "Analysis")
        
    def load_sample_data(self):
        """Load sample data for demonstration"""
        # Sample employees
        self.engine.add_employee(Employee(
            "emp001", "John Doe", 
            ["Python", "Machine Learning", "Data Analysis", "SQL"], 
            5, 
            ["AWS Certified", "PMI Agile"], 
            ["Become a ML architect", "Lead AI projects"],
            ["Project Alpha", "Project Beta"],
            "2023-10-01",
            {"location": "New York", "department": "AI"}
        ))
        
        self.engine.add_employee(Employee(
            "emp002", "Jane Smith", 
            ["Java", "Spring Boot", "Microservices", "AWS"], 
            3, 
            ["AWS Certified"], 
            ["Cloud architecture", "Lead engineering teams"],
            ["Project Gamma"],
            "2023-09-15",
            {"location": "Remote", "department": "Cloud"}
        ))
        
        # Sample roles
        self.engine.add_role(ProjectRole(
            "role001", "proj001", 
            "ML Engineer", 
            ["Python", "Machine Learning", "Data Analysis"], 
            ["TensorFlow", "PyTorch"],
            [],
            "2023-10-15",
            26,
            "New York",
            1
        ))
        
        self.engine.add_role(ProjectRole(
            "role002", "proj002", 
            "Cloud Architect", 
            ["AWS", "Microservices", "Java"], 
            ["Kubernetes", "Terraform"],
            ["AWS Certified"],
            "2023-11-01",
            52,
            "Remote",
            2
        ))
        
        self.refresh_employee_list()
        self.refresh_role_list()
        self.statusBar().showMessage("Sample data loaded", 3000)
        
    def add_employee(self):
        """Add a new employee from the form"""
        try:
            employee_id = self.emp_id_input.text().strip()
            name = self.emp_name_input.text().strip()
            skills = [s.strip() for s in self.emp_skills_input.text().split(",")]
            experience = int(self.emp_exp_input.text())
            certs = [c.strip() for c in self.emp_cert_input.text().split(",")]
            goals = [g.strip() for g in self.emp_goals_input.text().split(",")]
            avail_date = self.emp_avail_input.text().strip()
            
            # Parse preferences JSON
            pref_text = self.emp_pref_input.toPlainText().strip()
            preferences = json.loads(pref_text) if pref_text else {}
            
            # Create and add employee
            new_emp = Employee(
                employee_id, name, skills, experience, certs, 
                goals, [], avail_date, preferences
            )
            self.engine.add_employee(new_emp)
            
            # Clear form
            self.emp_id_input.clear()
            self.emp_name_input.clear()
            self.emp_skills_input.clear()
            self.emp_exp_input.clear()
            self.emp_cert_input.clear()
            self.emp_goals_input.clear()
            self.emp_avail_input.clear()
            self.emp_pref_input.clear()
            
            self.refresh_employee_list()
            self.statusBar().showMessage(f"Added employee: {name}", 3000)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to add employee: {str(e)}")
        
    def add_role(self):
        """Add a new role from the form"""
        try:
            role_id = self.role_id_input.text().strip()
            project_id = self.role_proj_input.text().strip()
            title = self.role_title_input.text().strip()
            req_skills = [s.strip() for s in self.role_req_skills_input.text().split(",")]
            nice_skills = [s.strip() for s in self.role_nice_skills_input.text().split(",")]
            certs = [c.strip() for c in self.role_cert_input.text().split(",")]
            start_date = self.role_start_input.text().strip()
            duration = int(self.role_dur_input.text())
            location = self.role_loc_input.text().strip()
            priority = int(self.role_pri_input.text())
            
            # Create and add role
            new_role = ProjectRole(
                role_id, project_id, title, req_skills, nice_skills,
                certs, start_date, duration, location, priority
            )
            self.engine.add_role(new_role)
            
            # Clear form
            self.role_id_input.clear()
            self.role_proj_input.clear()
            self.role_title_input.clear()
            self.role_req_skills_input.clear()
            self.role_nice_skills_input.clear()
            self.role_cert_input.clear()
            self.role_start_input.clear()
            self.role_dur_input.clear()
            self.role_loc_input.clear()
            self.role_pri_input.clear()
            
            self.refresh_role_list()
            self.statusBar().showMessage(f"Added role: {title}", 3000)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to add role: {str(e)}")
        
    def refresh_employee_list(self):
        """Refresh the employee list display"""
        self.employee_list.setRowCount(len(self.engine.employees))
        
        for row, emp in enumerate(self.engine.employees):
            self.employee_list.setItem(row, 0, QTableWidgetItem(emp.employee_id))
            self.employee_list.setItem(row, 1, QTableWidgetItem(emp.name))
            self.employee_list.setItem(row, 2, QTableWidgetItem(", ".join(emp.skills)))
            self.employee_list.setItem(row, 3, QTableWidgetItem(str(emp.experience_years)))
            
    def refresh_role_list(self):
        """Refresh the role list display"""
        self.role_list.setRowCount(len(self.engine.roles))
        
        for row, role in enumerate(self.engine.roles):
            self.role_list.setItem(row, 0, QTableWidgetItem(role.role_id))
            self.role_list.setItem(row, 1, QTableWidgetItem(role.title))
            self.role_list.setItem(row, 2, QTableWidgetItem(", ".join(role.required_skills)))
            self.role_list.setItem(row, 3, QTableWidgetItem(role.location))
            
    def build_matrix(self):
        """Build the skill matrix"""
        try:
            self.engine.build_skill_matrix()
            self.statusBar().showMessage("Skill matrix built successfully", 3000)
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to build skill matrix: {str(e)}")
        
    def find_matches(self):
        """Find matches based on user selection"""
        try:
            match_type = self.match_type_combo.currentText()
            entity_id = self.match_id_input.text().strip()
            
            if not entity_id:
                QMessageBox.warning(self, "Error", "Please enter an ID to match")
                return
                
            if match_type == "Employee to Roles":
                results = self.engine.match_employee_to_roles(entity_id)
                if not results:
                    QMessageBox.information(self, "No Matches", "No matching roles found for this employee")
                    return
                    
                self.match_results.setRowCount(len(results))
                self.match_results.setHorizontalHeaderLabels(["Role", "Score", "Details", "Compatibility"])
                
                for row, match in enumerate(results):
                    role = match['role']
                    self.match_results.setItem(row, 0, QTableWidgetItem(role['title']))
                    self.match_results.setItem(row, 1, QTableWidgetItem(f"{match['score']:.2f}"))
                    self.match_results.setItem(row, 2, QTableWidgetItem(match['match_details']))
                    
                    # Visual compatibility indicator
                    compat_item = QTableWidgetItem()
                    compat_item.setData(Qt.DisplayRole, "")
                    compat_item.setBackground(self.get_score_color(match['score']))
                    self.match_results.setItem(row, 3, compat_item)
                    
            else:  # Role to Employees
                results = self.engine.match_role_to_employees(entity_id)
                if not results:
                    QMessageBox.information(self, "No Matches", "No matching employees found for this role")
                    return
                    
                self.match_results.setRowCount(len(results))
                self.match_results.setHorizontalHeaderLabels(["Employee", "Score", "Details", "Compatibility"])
                
                for row, match in enumerate(results):
                    emp = match['employee']
                    self.match_results.setItem(row, 0, QTableWidgetItem(emp['name']))
                    self.match_results.setItem(row, 1, QTableWidgetItem(f"{match['score']:.2f}"))
                    self.match_results.setItem(row, 2, QTableWidgetItem(match['match_details']))
                    
                    # Visual compatibility indicator
                    compat_item = QTableWidgetItem()
                    compat_item.setData(Qt.DisplayRole, "")
                    compat_item.setBackground(self.get_score_color(match['score']))
                    self.match_results.setItem(row, 3, compat_item)
                    
            self.match_results.resizeColumnsToContents()
            self.statusBar().showMessage(f"Found {len(results)} matches", 3000)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to find matches: {str(e)}")
            
    def get_score_color(self, score):
        """Get color based on match score"""
        if score >= 0.8:
            return Qt.green
        elif score >= 0.6:
            return Qt.yellow
        else:
            return Qt.red
            
    def analyze_gaps(self):
        """Analyze skill gaps"""
        try:
            dept_filter = self.dept_filter_input.text().strip()
            dept_filter = dept_filter if dept_filter else None
            
            gaps = self.engine.analyze_skill_gaps(dept_filter)
            
            if not gaps:
                self.gap_analysis_results.setText("No significant skill gaps found")
                return
                
            report = "Skill Gap Analysis Report:\n\n"
            report += f"{'Skill':<30}{'Required':>10}{'Available':>10}{'Gap':>10}\n"
            report += "-" * 60 + "\n"
            
            for skill, data in gaps.items():
                report += f"{skill:<30}{data['required']:>10}{data['available']:>10}{data['gap']:>10}\n"
                
            self.gap_analysis_results.setText(report)
            self.statusBar().showMessage("Skill gap analysis completed", 3000)
            
        except Exception as e:
            QMessageBox.warning(self, "Error", f"Failed to analyze skill gaps: {str(e)}")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    app.setStyle('Fusion')  # Modern style
    
    # Set font
    font = QFont()
    font.setPointSize(10)
    app.setFont(font)
    
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
  
