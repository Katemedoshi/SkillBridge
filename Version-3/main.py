import sys
import json
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datetime import datetime, date
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
import logging
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Employee:
    """Employee dataclass with validation"""
    employee_id: str
    name: str
    skills: List[str]
    experience_years: int
    certifications: List[str]
    career_goals: List[str]
    availability_date: Union[str, date]
    preferences: Dict
    
    def __post_init__(self):
        if isinstance(self.availability_date, str):
            self.availability_date = datetime.strptime(self.availability_date, "%Y-%m-%d").date()
    
    def to_dict(self):
        return {
            "employee_id": self.employee_id,
            "name": self.name,
            "skills": self.skills,
            "experience_years": self.experience_years,
            "certifications": self.certifications,
            "career_goals": self.career_goals,
            "availability_date": self.availability_date.strftime("%Y-%m-%d"),
            "preferences": self.preferences
        }

@dataclass
class ProjectRole:
    """Project role dataclass with validation"""
    role_id: str
    title: str
    required_skills: List[str]
    nice_to_have_skills: List[str]
    required_certifications: List[str]
    start_date: Union[str, date]
    duration_weeks: int
    location: str
    
    def __post_init__(self):
        if isinstance(self.start_date, str):
            self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d").date()
    
    def to_dict(self):
        return {
            "role_id": self.role_id,
            "title": self.title,
            "required_skills": self.required_skills,
            "nice_to_have_skills": self.nice_to_have_skills,
            "required_certifications": self.required_certifications,
            "start_date": self.start_date.strftime("%Y-%m-%d"),
            "duration_weeks": self.duration_weeks,
            "location": self.location
        }

class AIMatchingEngine:
    """Efficient AI matching engine with caching"""
    
    def __init__(self):
        self.skill_vectorizer = TfidfVectorizer(stop_words='english')
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.employees = []
        self.roles = []
        self.skill_matrix = None
        self._cache = {}
    
    def add_employee(self, employee: Employee):
        """Add employee with duplicate check"""
        if any(emp.employee_id == employee.employee_id for emp in self.employees):
            return False
        self.employees.append(employee)
        self._invalidate_cache()
        return True
    
    def add_role(self, role: ProjectRole):
        """Add role with duplicate check"""
        if any(r.role_id == role.role_id for r in self.roles):
            return False
        self.roles.append(role)
        self._invalidate_cache()
        return True
    
    def _invalidate_cache(self):
        """Clear cache when data changes"""
        self._cache.clear()
        self.skill_matrix = None
    
    def build_skill_matrix(self):
        """Build optimized skill matrix"""
        if self.skill_matrix is not None:
            return
            
        all_texts = []
        for emp in self.employees:
            all_texts.append(" ".join(emp.skills))
        for role in self.roles:
            all_texts.append(" ".join(role.required_skills + role.nice_to_have_skills))
            
        if all_texts:
            self.skill_matrix = self.skill_vectorizer.fit_transform(all_texts)
    
    def _calculate_match_score(self, employee: Employee, role: ProjectRole) -> Dict:
        """Calculate comprehensive match score"""
        # Availability check
        if role.start_date < employee.availability_date:
            return {"score": 0, "reason": "Unavailable"}
        
        # Skill matching
        emp_skills_text = " ".join(employee.skills)
        role_skills_text = " ".join(role.required_skills)
        
        if self.skill_matrix is not None:
            emp_idx = next((i for i, emp in enumerate(self.employees) if emp.employee_id == employee.employee_id), -1)
            if emp_idx >= 0:
                emp_vec = self.skill_matrix[emp_idx]
                role_vec = self.skill_vectorizer.transform([role_skills_text])
                skill_score = cosine_similarity(emp_vec, role_vec)[0][0]
            else:
                emp_vec = self.skill_vectorizer.transform([emp_skills_text])
                role_vec = self.skill_vectorizer.transform([role_skills_text])
                skill_score = cosine_similarity(emp_vec, role_vec)[0][0]
        else:
            skill_score = 0.5
        
        # Certification matching
        cert_score = len(set(employee.certifications) & set(role.required_certifications)) / max(1, len(role.required_certifications))
        
        # Career alignment
        career_score = 0
        if employee.career_goals:
            career_text = " ".join(employee.career_goals)
            role_text = f"{role.title} {role_skills_text}"
            career_score = self._get_semantic_similarity(career_text, role_text)
        
        # Location preference
        pref_location = employee.preferences.get('location', '')
        location_score = 1 if (not pref_location or pref_location == role.location or role.location == "Remote") else 0.3
        
        # Weighted final score
        final_score = (0.5 * skill_score + 0.2 * cert_score + 
                      0.2 * career_score + 0.1 * location_score)
        
        return {
            "score": final_score,
            "skill_match": skill_score,
            "cert_match": cert_score,
            "career_alignment": career_score,
            "location_match": location_score
        }
    
    def _get_semantic_similarity(self, text1: str, text2: str) -> float:
        """Cached semantic similarity calculation"""
        cache_key = hash((text1, text2))
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        embeddings = self.sbert_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        self._cache[cache_key] = similarity
        return similarity
    
    def match_employee_to_roles(self, employee_id: str, top_n: int = 5) -> List[Dict]:
        """Find matching roles for employee"""
        employee = next((emp for emp in self.employees if emp.employee_id == employee_id), None)
        if not employee:
            return []
        
        results = []
        for role in self.roles:
            match_data = self._calculate_match_score(employee, role)
            if match_data["score"] > 0:
                results.append({
                    "role": role.to_dict(),
                    **match_data
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
    
    def match_role_to_employees(self, role_id: str, top_n: int = 5) -> List[Dict]:
        """Find matching employees for role"""
        role = next((r for r in self.roles if r.role_id == role_id), None)
        if not role:
            return []
        
        results = []
        for employee in self.employees:
            match_data = self._calculate_match_score(employee, role)
            if match_data["score"] > 0:
                results.append({
                    "employee": employee.to_dict(),
                    **match_data
                })
        
        return sorted(results, key=lambda x: x['score'], reverse=True)[:top_n]
    
    def analyze_skill_gaps(self, department: str = None) -> Dict:
        """Analyze skill gaps in the organization"""
        required_skills = []
        for role in self.roles:
            required_skills.extend(role.required_skills)
        
        available_skills = []
        for emp in self.employees:
            if not department or emp.preferences.get('department') == department:
                available_skills.extend(emp.skills)
        
        required_counts = {}
        for skill in required_skills:
            required_counts[skill] = required_counts.get(skill, 0) + 1
        
        available_counts = {}
        for skill in available_skills:
            available_counts[skill] = available_counts.get(skill, 0) + 1
        
        gaps = {}
        for skill, required in required_counts.items():
            available = available_counts.get(skill, 0)
            if required > available:
                gaps[skill] = {
                    "required": required,
                    "available": available,
                    "gap": required - available
                }
        
        return dict(sorted(gaps.items(), key=lambda x: x[1]['gap'], reverse=True))

class ModernUI(QMainWindow):
    """Modern and efficient UI for the matching system"""
    
    def __init__(self):
        super().__init__()
        self.engine = AIMatchingEngine()
        self.setup_ui()
        self.load_sample_data()
        self.setWindowTitle("AI Talent Matching System")
        self.resize(1000, 700)
    
    def setup_ui(self):
        """Setup the main UI components"""
        self.setStyleSheet(self._get_stylesheet())
        
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(15, 15, 15, 15)
        
        # Header
        header = QLabel("AI Talent Matching System")
        header.setStyleSheet("font-size: 20px; font-weight: bold; color: #2c3e50;")
        layout.addWidget(header)
        
        # Main tabs
        self.tabs = QTabWidget()
        layout.addWidget(self.tabs)
        
        # Create tabs
        self._create_employee_tab()
        self._create_role_tab()
        self._create_matching_tab()
        self._create_analysis_tab()
        
        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready")
    
    def _create_employee_tab(self):
        """Create employee management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Form
        form_group = QGroupBox("Add Employee")
        form_layout = QFormLayout(form_group)
        
        self.emp_id = QLineEdit()
        self.emp_name = QLineEdit()
        self.emp_skills = QLineEdit()
        self.emp_exp = QLineEdit()
        self.emp_certs = QLineEdit()
        self.emp_goals = QLineEdit()
        self.emp_avail = QLineEdit()
        self.emp_prefs = QTextEdit()
        self.emp_prefs.setMaximumHeight(60)
        
        form_layout.addRow("ID*:", self.emp_id)
        form_layout.addRow("Name*:", self.emp_name)
        form_layout.addRow("Skills:", self.emp_skills)
        form_layout.addRow("Experience:", self.emp_exp)
        form_layout.addRow("Certifications:", self.emp_certs)
        form_layout.addRow("Career Goals:", self.emp_goals)
        form_layout.addRow("Availability:", self.emp_avail)
        form_layout.addRow("Preferences:", self.emp_prefs)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Employee")
        add_btn.clicked.connect(self._add_employee)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_employee_form)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(clear_btn)
        form_layout.addRow(btn_layout)
        
        # Employee list
        list_group = QGroupBox("Employees")
        list_layout = QVBoxLayout(list_group)
        
        self.employee_table = QTableWidget()
        self.employee_table.setColumnCount(4)
        self.employee_table.setHorizontalHeaderLabels(["ID", "Name", "Skills", "Experience"])
        self.employee_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        list_layout.addWidget(self.employee_table)
        
        # Add to tab
        layout.addWidget(form_group)
        layout.addWidget(list_group)
        self.tabs.addTab(tab, "Employees")
    
    def _create_role_tab(self):
        """Create role management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Form
        form_group = QGroupBox("Add Role")
        form_layout = QFormLayout(form_group)
        
        self.role_id = QLineEdit()
        self.role_title = QLineEdit()
        self.role_skills = QLineEdit()
        self.role_certs = QLineEdit()
        self.role_start = QLineEdit()
        self.role_duration = QLineEdit()
        self.role_location = QLineEdit()
        
        form_layout.addRow("ID*:", self.role_id)
        form_layout.addRow("Title*:", self.role_title)
        form_layout.addRow("Required Skills:", self.role_skills)
        form_layout.addRow("Required Certs:", self.role_certs)
        form_layout.addRow("Start Date:", self.role_start)
        form_layout.addRow("Duration (weeks):", self.role_duration)
        form_layout.addRow("Location:", self.role_location)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = QPushButton("Add Role")
        add_btn.clicked.connect(self._add_role)
        clear_btn = QPushButton("Clear")
        clear_btn.clicked.connect(self._clear_role_form)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(clear_btn)
        form_layout.addRow(btn_layout)
        
        # Role list
        list_group = QGroupBox("Roles")
        list_layout = QVBoxLayout(list_group)
        
        self.role_table = QTableWidget()
        self.role_table.setColumnCount(3)
        self.role_table.setHorizontalHeaderLabels(["ID", "Title", "Skills"])
        self.role_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        list_layout.addWidget(self.role_table)
        
        # Add to tab
        layout.addWidget(form_group)
        layout.addWidget(list_group)
        self.tabs.addTab(tab, "Roles")
    
    def _create_matching_tab(self):
        """Create matching tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_group = QGroupBox("Matching Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.match_type = QComboBox()
        self.match_type.addItems(["Employee to Roles", "Role to Employees"])
        self.match_id = QLineEdit()
        self.match_id.setPlaceholderText("Enter ID")
        
        match_btn = QPushButton("Find Matches")
        match_btn.clicked.connect(self._find_matches)
        
        controls_layout.addWidget(QLabel("Match Type:"))
        controls_layout.addWidget(self.match_type)
        controls_layout.addWidget(QLabel("ID:"))
        controls_layout.addWidget(self.match_id)
        controls_layout.addWidget(match_btn)
        
        # Results
        results_group = QGroupBox("Match Results")
        results_layout = QVBoxLayout(results_group)
        
        self.match_table = QTableWidget()
        self.match_table.setColumnCount(4)
        self.match_table.setHorizontalHeaderLabels(["Match", "Score", "Skills", "Details"])
        self.match_table.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        
        results_layout.addWidget(self.match_table)
        
        # Add to tab
        layout.addWidget(controls_group)
        layout.addWidget(results_group)
        self.tabs.addTab(tab, "Matching")
    
    def _create_analysis_tab(self):
        """Create analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_group = QGroupBox("Analysis Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        self.dept_filter = QLineEdit()
        self.dept_filter.setPlaceholderText("Department (optional)")
        
        analyze_btn = QPushButton("Analyze Skill Gaps")
        analyze_btn.clicked.connect(self._analyze_gaps)
        
        controls_layout.addWidget(QLabel("Filter:"))
        controls_layout.addWidget(self.dept_filter)
        controls_layout.addWidget(analyze_btn)
        
        # Results
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.analysis_text = QTextEdit()
        self.analysis_text.setReadOnly(True)
        self.analysis_text.setFont(QFont("Courier", 10))
        
        results_layout.addWidget(self.analysis_text)
        
        # Add to tab
        layout.addWidget(controls_group)
        layout.addWidget(results_group)
        self.tabs.addTab(tab, "Analysis")
    
    def _get_stylesheet(self):
        """Return modern stylesheet"""
        return """
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                border: 1px solid #ddd;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLineEdit, QTextEdit, QComboBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 5px;
            }
            QPushButton {
                background-color: #4a9fdc;
                color: white;
                border: none;
                border-radius: 4px;
                padding: 5px 10px;
            }
            QPushButton:hover {
                background-color: #3a8ccc;
            }
            QTableWidget {
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            QTabWidget::pane {
                border: 1px solid #ddd;
                border-radius: 5px;
            }
            QTabBar::tab {
                padding: 8px 15px;
                border-top-left-radius: 5px;
                border-top-right-radius: 5px;
            }
            QTabBar::tab:selected {
                background: #f8f9fa;
                border-bottom: 2px solid #4a9fdc;
            }
        """
    
    def _add_employee(self):
        """Add employee from form"""
        try:
            data = {
                "id": self.emp_id.text().strip(),
                "name": self.emp_name.text().strip(),
                "skills": [s.strip() for s in self.emp_skills.text().split(",") if s.strip()],
                "experience": int(self.emp_exp.text()) if self.emp_exp.text() else 0,
                "certs": [c.strip() for c in self.emp_certs.text().split(",") if c.strip()],
                "goals": [g.strip() for g in self.emp_goals.text().split(",") if g.strip()],
                "date": self.emp_avail.text().strip() or datetime.now().strftime("%Y-%m-%d"),
                "prefs": self.emp_prefs.toPlainText().strip()
            }
            
            if not data["id"] or not data["name"]:
                self.status_bar.showMessage("ID and Name are required", 3000)
                return
            
            try:
                prefs = json.loads(data["prefs"]) if data["prefs"] else {}
            except json.JSONDecodeError:
                self.status_bar.showMessage("Invalid JSON in preferences", 3000)
                return
            
            employee = Employee(
                employee_id=data["id"],
                name=data["name"],
                skills=data["skills"],
                experience_years=data["experience"],
                certifications=data["certs"],
                career_goals=data["goals"],
                availability_date=data["date"],
                preferences=prefs
            )
            
            if self.engine.add_employee(employee):
                self.status_bar.showMessage(f"Added employee {data['name']}", 3000)
                self._clear_employee_form()
                self._refresh_employee_list()
            else:
                self.status_bar.showMessage(f"Employee {data['id']} already exists", 3000)
                
        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}", 3000)
    
    def _add_role(self):
        """Add role from form"""
        try:
            data = {
                "id": self.role_id.text().strip(),
                "title": self.role_title.text().strip(),
                "skills": [s.strip() for s in self.role_skills.text().split(",") if s.strip()],
                "certs": [c.strip() for c in self.role_certs.text().split(",") if c.strip()],
                "start": self.role_start.text().strip() or datetime.now().strftime("%Y-%m-%d"),
                "duration": int(self.role_duration.text()) if self.role_duration.text() else 4,
                "location": self.role_location.text().strip() or "Remote"
            }
            
            if not data["id"] or not data["title"]:
                self.status_bar.showMessage("ID and Title are required", 3000)
                return
            
            role = ProjectRole(
                role_id=data["id"],
                title=data["title"],
                required_skills=data["skills"],
                nice_to_have_skills=[],
                required_certifications=data["certs"],
                start_date=data["start"],
                duration_weeks=data["duration"],
                location=data["location"]
            )
            
            if self.engine.add_role(role):
                self.status_bar.showMessage(f"Added role {data['title']}", 3000)
                self._clear_role_form()
                self._refresh_role_list()
            else:
                self.status_bar.showMessage(f"Role {data['id']} already exists", 3000)
                
        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}", 3000)
    
    def _clear_employee_form(self):
        """Clear employee form"""
        self.emp_id.clear()
        self.emp_name.clear()
        self.emp_skills.clear()
        self.emp_exp.clear()
        self.emp_certs.clear()
        self.emp_goals.clear()
        self.emp_avail.clear()
        self.emp_prefs.clear()
    
    def _clear_role_form(self):
        """Clear role form"""
        self.role_id.clear()
        self.role_title.clear()
        self.role_skills.clear()
        self.role_certs.clear()
        self.role_start.clear()
        self.role_duration.clear()
        self.role_location.clear()
    
    def _refresh_employee_list(self):
        """Refresh employee table"""
        self.employee_table.setRowCount(len(self.engine.employees))
        
        for row, emp in enumerate(self.engine.employees):
            self.employee_table.setItem(row, 0, QTableWidgetItem(emp.employee_id))
            self.employee_table.setItem(row, 1, QTableWidgetItem(emp.name))
            self.employee_table.setItem(row, 2, QTableWidgetItem(", ".join(emp.skills[:3])))
            self.employee_table.setItem(row, 3, QTableWidgetItem(str(emp.experience_years)))
    
    def _refresh_role_list(self):
        """Refresh role table"""
        self.role_table.setRowCount(len(self.engine.roles))
        
        for row, role in enumerate(self.engine.roles):
            self.role_table.setItem(row, 0, QTableWidgetItem(role.role_id))
            self.role_table.setItem(row, 1, QTableWidgetItem(role.title))
            self.role_table.setItem(row, 2, QTableWidgetItem(", ".join(role.required_skills[:3])))
    
    def _find_matches(self):
        """Find and display matches"""
        try:
            match_type = self.match_type.currentText()
            entity_id = self.match_id.text().strip()
            
            if not entity_id:
                self.status_bar.showMessage("Please enter an ID", 3000)
                return
            
            if match_type == "Employee to Roles":
                results = self.engine.match_employee_to_roles(entity_id)
                self._display_matches(results, "role")
            else:
                results = self.engine.match_role_to_employees(entity_id)
                self._display_matches(results, "employee")
            
            self.status_bar.showMessage(f"Found {len(results)} matches", 3000)
            
        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}", 3000)
    
    def _display_matches(self, results, match_type):
        """Display match results in table"""
        self.match_table.setRowCount(len(results))
        
        for row, match in enumerate(results):
            entity = match[match_type]
            
            name_item = QTableWidgetItem(entity["name"] if match_type == "employee" else entity["title"])
            score_item = QTableWidgetItem(f"{match['score']:.2f}")
            skills_item = QTableWidgetItem(", ".join(entity["skills"][:3] if match_type == "employee" else entity["required_skills"][:3]))
            details_item = QTableWidgetItem(f"Skills: {match['skill_match']:.2f}, Certs: {match['cert_match']:.2f}")
            
            # Color coding
            score = match['score']
            if score >= 0.8:
                color = QColor(220, 255, 220)  # Light green
            elif score >= 0.6:
                color = QColor(255, 255, 200)  # Light yellow
            elif score >= 0.4:
                color = QColor(255, 230, 200)  # Light orange
            else:
                color = QColor(255, 220, 220)  # Light red
            
            for item in [name_item, score_item, skills_item, details_item]:
                item.setBackground(color)
            
            self.match_table.setItem(row, 0, name_item)
            self.match_table.setItem(row, 1, score_item)
            self.match_table.setItem(row, 2, skills_item)
            self.match_table.setItem(row, 3, details_item)
    
    def _analyze_gaps(self):
        """Analyze and display skill gaps"""
        try:
            dept = self.dept_filter.text().strip() or None
            gaps = self.engine.analyze_skill_gaps(dept)
            
            if not gaps:
                self.analysis_text.setText("No significant skill gaps found")
                return
            
            report = "Skill Gap Analysis Report\n"
            report += "=" * 30 + "\n\n"
            
            if dept:
                report += f"Department: {dept}\n\n"
            
            report += f"{'Skill':<20}{'Required':>10}{'Available':>10}{'Gap':>10}\n"
            report += "-" * 50 + "\n"
            
            for skill, data in list(gaps.items())[:20]:  # Show top 20
                report += f"{skill[:18]:<20}{data['required']:>10}{data['available']:>10}{data['gap']:>10}\n"
            
            self.analysis_text.setText(report)
            self.status_bar.showMessage(f"Found {len(gaps)} skill gaps", 3000)
            
        except Exception as e:
            self.status_bar.showMessage(f"Error: {str(e)}", 3000)
    
    def load_sample_data(self):
        """Load sample data for demonstration"""
        try:
            # Sample employees
            employees = [
                Employee("e001", "John Smith", 
                        ["Python", "ML", "Data Analysis", "SQL"], 
                        5, ["AWS Certified"], 
                        ["Become ML expert"], 
                        "2023-10-01", 
                        {"location": "NYC"}),
                
                Employee("e002", "Jane Doe", 
                        ["Java", "Spring", "Microservices"], 
                        3, [], 
                        ["Cloud architecture"], 
                        "2023-09-15", 
                        {"location": "Remote"}),
                
                Employee("e003", "Mike Johnson", 
                        ["JavaScript", "React", "Node.js"], 
                        4, [], 
                        ["Full-stack development"], 
                        "2023-11-01", 
                        {"location": "SF"})
            ]
            
            # Sample roles
            roles = [
                ProjectRole("r001", "ML Engineer",
                          ["Python", "ML", "Data Analysis"], 
                          ["TensorFlow"], ["AWS Certified"], 
                          "2023-10-15", 26, "NYC"),
                
                ProjectRole("r002", "Java Developer", 
                          ["Java", "Spring"], 
                          ["Microservices"], [], 
                          "2023-11-01", 12, "Remote"),
                
                ProjectRole("r003", "Frontend Developer",
                          ["JavaScript", "React"], 
                          ["Node.js"], [], 
                          "2023-12-01", 8, "SF")
            ]
            
            # Add to engine
            for emp in employees:
                self.engine.add_employee(emp)
            
            for role in roles:
                self.engine.add_role(role)
            
            # Refresh UI
            self._refresh_employee_list()
            self._refresh_role_list()
            
            self.status_bar.showMessage("Loaded sample data", 3000)
            
        except Exception as e:
            self.status_bar.showMessage(f"Error loading sample data: {str(e)}", 3000)

def main():
    """Application entry point"""
    app = QApplication(sys.argv)
    
    # Set modern font
    font = QFont()
    font.setFamily("Segoe UI")
    font.setPointSize(10)
    app.setFont(font)
    
    # Create and show window
    window = ModernUI()
    window.show()
    
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
