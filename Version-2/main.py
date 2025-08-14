import sys
import json
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sentence_transformers import SentenceTransformer
from datetime import datetime, date
from typing import List, Dict, Optional, Union
from dataclasses import dataclass, asdict
import logging
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import seaborn as sns

from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class Employee:
    """Enhanced Employee dataclass with validation"""
    employee_id: str
    name: str
    skills: List[str]
    experience_years: int
    certifications: List[str]
    career_goals: List[str]
    past_projects: List[str]
    availability_date: Union[str, date]
    preferences: Dict
    
    def __post_init__(self):
        if isinstance(self.availability_date, str):
            self.availability_date = datetime.strptime(self.availability_date, "%Y-%m-%d").date()
    
    def to_dict(self):
        data = asdict(self)
        data['availability_date'] = self.availability_date.strftime("%Y-%m-%d")
        return data

@dataclass
class ProjectRole:
    """Enhanced ProjectRole dataclass with validation"""
    role_id: str
    project_id: str
    title: str
    required_skills: List[str]
    nice_to_have_skills: List[str]
    required_certifications: List[str]
    start_date: Union[str, date]
    duration_weeks: int
    location: str
    priority: int
    
    def __post_init__(self):
        if isinstance(self.start_date, str):
            self.start_date = datetime.strptime(self.start_date, "%Y-%m-%d").date()
    
    def to_dict(self):
        data = asdict(self)
        data['start_date'] = self.start_date.strftime("%Y-%m-%d")
        return data

class AIMatchingEngine:
    """Streamlined AI matching engine with caching and performance optimization"""
    
    def __init__(self):
        self.skill_vectorizer = TfidfVectorizer(stop_words='english', max_features=1000)
        self.sbert_model = SentenceTransformer('all-MiniLM-L6-v2')
        self.employees: List[Employee] = []
        self.roles: List[ProjectRole] = []
        self.skill_matrix = None
        self._cache = {}
        
    def add_employee(self, employee: Employee) -> bool:
        """Add employee with duplicate checking"""
        if any(emp.employee_id == employee.employee_id for emp in self.employees):
            logger.warning(f"Employee {employee.employee_id} already exists")
            return False
        self.employees.append(employee)
        self._invalidate_cache()
        return True
        
    def add_role(self, role: ProjectRole) -> bool:
        """Add role with duplicate checking"""
        if any(r.role_id == role.role_id for r in self.roles):
            logger.warning(f"Role {role.role_id} already exists")
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
            logger.info("Skill matrix built successfully")
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Cached semantic similarity calculation"""
        cache_key = hash((text1, text2))
        if cache_key in self._cache:
            return self._cache[cache_key]
            
        embeddings = self.sbert_model.encode([text1, text2])
        similarity = cosine_similarity([embeddings[0]], [embeddings[1]])[0][0]
        self._cache[cache_key] = similarity
        return similarity
    
    def _calculate_match_score(self, employee: Employee, role: ProjectRole) -> Dict:
        """Calculate comprehensive match score"""
        # Availability check
        if role.start_date < employee.availability_date:
            return {"score": 0, "reason": "Availability mismatch"}
        
        # Skill matching
        emp_skills_text = " ".join(employee.skills)
        role_skills_text = " ".join(role.required_skills)
        
        if self.skill_matrix is not None:
            emp_idx = len(self.employees) - 1 if employee in self.employees else -1
            if emp_idx >= 0:
                emp_vec = self.skill_matrix[emp_idx]
                role_vec = self.skill_vectorizer.transform([role_skills_text])
                skill_score = cosine_similarity(emp_vec, role_vec)[0][0]
            else:
                emp_vec = self.skill_vectorizer.transform([emp_skills_text])
                role_vec = self.skill_vectorizer.transform([role_skills_text])
                skill_score = cosine_similarity(emp_vec, role_vec)[0][0]
        else:
            skill_score = 0.5  # Default if matrix not built
        
        # Certification matching
        emp_certs = set(employee.certifications)
        req_certs = set(role.required_certifications)
        cert_score = len(emp_certs & req_certs) / max(1, len(req_certs))
        
        # Career alignment
        career_score = 0
        if employee.career_goals:
            career_text = " ".join(employee.career_goals)
            role_text = f"{role.title} {role_skills_text}"
            career_score = self._calculate_semantic_similarity(career_text, role_text)
        
        # Location preference
        location_score = 1 if (not employee.preferences.get('location') or 
                             employee.preferences['location'] == role.location or
                             role.location == "Remote") else 0.3
        
        # Weighted final score
        final_score = (0.5 * skill_score + 0.2 * cert_score + 
                      0.2 * career_score + 0.1 * location_score)
        
        return {
            "score": final_score,
            "skill_match": skill_score,
            "cert_match": cert_score,
            "career_alignment": career_score,
            "location_match": location_score,
            "details": f"Skills: {skill_score:.2f}, Certs: {cert_score:.2f}, Career: {career_score:.2f}"
        }
    
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
        """Optimized skill gap analysis"""
        from collections import Counter
        
        # Count required skills
        required_skills = []
        for role in self.roles:
            required_skills.extend(role.required_skills)
        
        # Count available skills
        available_skills = []
        for emp in self.employees:
            if not department or emp.preferences.get('department') == department:
                available_skills.extend(emp.skills)
        
        required_counts = Counter(required_skills)
        available_counts = Counter(available_skills)
        
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

class StatusMixin:
    """Mixin for status updates"""
    
    def show_status(self, message: str, status_type: str = "info"):
        """Show status message with color coding"""
        colors = {
            "info": "#17a2b8",
            "success": "#28a745", 
            "warning": "#ffc107",
            "error": "#dc3545",
            "processing": "#6f42c1"
        }
        
        if hasattr(self, 'status_label'):
            self.status_label.setText(message)
            self.status_label.setStyleSheet(f"color: {colors.get(status_type, '#17a2b8')}; font-weight: bold;")
        
        logger.info(f"{status_type.upper()}: {message}")

class FormHelper:
    """Helper class for form creation"""
    
    @staticmethod
    def create_form_row(layout, label_text: str, widget: QWidget, required: bool = False):
        """Create consistent form rows"""
        row = QHBoxLayout()
        
        label = QLabel(label_text + ("*" if required else ""))
        label.setMinimumWidth(150)
        if required:
            label.setStyleSheet("font-weight: bold; color: #dc3545;")
        
        row.addWidget(label)
        row.addWidget(widget, 1)
        layout.addLayout(row)
        return widget
    
    @staticmethod
    def validate_form_data(data: Dict) -> tuple:
        """Validate form data"""
        errors = []
        
        # Required field validation
        required_fields = ['id', 'name']
        for field in required_fields:
            if not data.get(field, '').strip():
                errors.append(f"{field.title()} is required")
        
        # Date validation
        if 'date' in data and data['date']:
            try:
                datetime.strptime(data['date'], "%Y-%m-%d")
            except ValueError:
                errors.append("Invalid date format (use YYYY-MM-DD)")
        
        # Numeric validation
        for field in ['experience', 'duration', 'priority']:
            if field in data and data[field]:
                try:
                    int(data[field])
                except ValueError:
                    errors.append(f"{field.title()} must be a number")
        
        return len(errors) == 0, errors

class ChartWidget(QWidget):
    """Custom chart widget using matplotlib instead of PyQt5.QtChart"""
    
    def __init__(self, parent=None):
        super().__init__(parent)
        self.figure = Figure(figsize=(10, 6))
        self.canvas = FigureCanvas(self.figure)
        
        layout = QVBoxLayout()
        layout.addWidget(self.canvas)
        self.setLayout(layout)
        
        # Set style
        plt.style.use('seaborn-v0_8' if 'seaborn-v0_8' in plt.style.available else 'default')
        
    def create_bar_chart(self, labels, values, title="Chart", xlabel="Categories", ylabel="Values", colors=None):
        """Create a bar chart"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        if colors is None:
            colors = plt.cm.Set3(np.linspace(0, 1, len(labels)))
        
        bars = ax.bar(labels, values, color=colors)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel(xlabel, fontsize=12)
        ax.set_ylabel(ylabel, fontsize=12)
        
        # Rotate labels if they're too long
        max_label_length = max(len(str(label)) for label in labels) if labels else 0
        if max_label_length > 10:
            plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        
        # Add value labels on bars
        for bar, value in zip(bars, values):
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.1f}' if isinstance(value, float) else str(value),
                   ha='center', va='bottom', fontweight='bold')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def create_horizontal_bar_chart(self, labels, values, title="Chart"):
        """Create a horizontal bar chart for better label visibility"""
        self.figure.clear()
        ax = self.figure.add_subplot(111)
        
        # Create color map based on values
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(values)))
        
        bars = ax.barh(labels, values, color=colors)
        ax.set_title(title, fontsize=14, fontweight='bold', pad=20)
        ax.set_xlabel('Gap Size', fontsize=12)
        
        # Add value labels
        for bar, value in zip(bars, values):
            width = bar.get_width()
            ax.text(width + 0.1, bar.get_y() + bar.get_height()/2.,
                   str(value), ha='left', va='center', fontweight='bold')
        
        self.figure.tight_layout()
        self.canvas.draw()
    
    def create_match_visualization(self, matches, match_type="Matches"):
        """Create visualization for match results"""
        if not matches:
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, 'No matches found', ha='center', va='center',
                   fontsize=16, transform=ax.transAxes)
            ax.set_title("Match Results", fontsize=14, fontweight='bold')
            self.canvas.draw()
            return
        
        # Extract data
        labels = []
        scores = []
        
        for match in matches:
            if match_type == "Roles":
                labels.append(match['role']['title'][:20] + "..." if len(match['role']['title']) > 20 else match['role']['title'])
            else:
                labels.append(match['employee']['name'])
            scores.append(match['score'] * 100)  # Convert to percentage
        
        # Create color map based on scores
        colors = ['#28a745' if score >= 80 else '#ffc107' if score >= 60 else '#dc3545' for score in scores]
        
        self.create_bar_chart(labels, scores, 
                             f"Top {len(matches)} Matching {match_type}",
                             match_type, "Match Score (%)", colors)

class ProgressWidget(QWidget):
    """Custom progress widget with color coding"""
    
    def __init__(self, value=0, parent=None):
        super().__init__(parent)
        self.value = max(0, min(100, value))
        self.setMinimumSize(100, 20)
        self.setMaximumHeight(20)
    
    def setValue(self, value):
        self.value = max(0, min(100, value))
        self.update()
    
    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        
        # Background
        painter.setBrush(QBrush(QColor(240, 240, 240)))
        painter.setPen(QPen(QColor(200, 200, 200), 1))
        painter.drawRoundedRect(self.rect(), 3, 3)
        
        # Progress bar
        if self.value > 0:
            progress_width = int(self.width() * self.value / 100)
            
            # Color based on value
            if self.value >= 80:
                color = QColor(40, 167, 69)  # Green
            elif self.value >= 60:
                color = QColor(255, 193, 7)   # Yellow
            else:
                color = QColor(220, 53, 69)   # Red
            
            painter.setBrush(QBrush(color))
            painter.setPen(QPen(color, 1))
            painter.drawRoundedRect(0, 0, progress_width, self.height(), 3, 3)
        
        # Text
        painter.setPen(QPen(QColor(50, 50, 50)))
        painter.drawText(self.rect(), Qt.AlignCenter, f"{self.value:.0f}%")

class ModernButton(QPushButton):
    """Enhanced button with modern styling and animations"""
    
    def __init__(self, text: str, icon: str = None, style_type: str = "primary"):
        super().__init__(text)
        self.style_type = style_type
        self.setup_style()
        
        if icon:
            self.setText(f"{icon} {text}")
    
    def setup_style(self):
        styles = {
            "primary": "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #4a9fdc, stop:1 #3a7fc4);",
            "success": "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #28a745, stop:1 #218838);",
            "warning": "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #ffc107, stop:1 #e0a800); color: #212529;",
            "danger": "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #dc3545, stop:1 #c82333);",
            "secondary": "background: qlineargradient(x1:0, y1:0, x2:0, y2:1, stop:0 #6c757d, stop:1 #5a6268);"
        }
        
        base_style = """
            QPushButton {
                color: white;
                border: 1px solid #2d5d7d;
                border-radius: 6px;
                padding: 8px 16px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                transform: translateY(-2px);
                box-shadow: 0 4px 8px rgba(0,0,0,0.2);
            }
            QPushButton:pressed {
                transform: translateY(0);
            }
        """
        
        self.setStyleSheet(base_style + styles.get(self.style_type, styles["primary"]))

class EnhancedTableWidget(QTableWidget):
    """Enhanced table with modern styling and features"""
    
    def __init__(self, columns: List[str]):
        super().__init__()
        self.setColumnCount(len(columns))
        self.setHorizontalHeaderLabels(columns)
        self.setup_style()
        
    def setup_style(self):
        self.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        self.setSelectionBehavior(QAbstractItemView.SelectRows)
        self.setAlternatingRowColors(True)
        self.verticalHeader().setVisible(False)
        
        self.setStyleSheet("""
            QTableWidget {
                border: 1px solid #dee2e6;
                border-radius: 8px;
                background: white;
                gridline-color: #e9ecef;
            }
            QTableWidget::item {
                padding: 8px;
                border-bottom: 1px solid #e9ecef;
            }
            QTableWidget::item:selected {
                background: #3a7fc4;
                color: white;
            }
            QHeaderView::section {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1, 
                                          stop:0 #4a9fdc, stop:1 #3a7fc4);
                color: white;
                padding: 10px;
                border: none;
                font-weight: bold;
            }
        """)

class MainWindow(QMainWindow, StatusMixin):
    """Streamlined main application window"""
    
    def __init__(self):
        super().__init__()
        self.engine = AIMatchingEngine()
        self.setup_ui()
        self.load_sample_data()
        
    def setup_ui(self):
        """Setup modern UI with improved organization"""
        self.setWindowTitle("🚀 AI Talent Matching System v2.0")
        self.setGeometry(100, 100, 1200, 800)
        self.setStyleSheet(self.get_app_stylesheet())
        
        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)
        layout.setContentsMargins(20, 20, 20, 20)
        
        # Header
        header = self.create_header()
        layout.addWidget(header)
        
        # Main tabs
        self.tabs = QTabWidget()
        self.tabs.setStyleSheet(self.get_tab_stylesheet())
        layout.addWidget(self.tabs)
        
        # Create tabs
        self.create_employee_tab()
        self.create_role_tab()
        self.create_matching_tab()
        self.create_analysis_tab()
        
        # Status bar
        self.statusBar().showMessage("Ready")
        
    def create_header(self) -> QWidget:
        """Create application header"""
        header = QWidget()
        layout = QHBoxLayout(header)
        
        # Logo and title
        title_layout = QHBoxLayout()
        logo = QLabel("🤖")
        logo.setFont(QFont("Arial", 20))
        title_layout.addWidget(logo)
        
        title = QLabel("AI Talent Matching System")
        title.setStyleSheet("font-size: 18px; font-weight: bold; color: #2c3e50;")
        title_layout.addWidget(title)
        layout.addLayout(title_layout)
        
        layout.addStretch()
        
        # Status label
        self.status_label = QLabel("System Ready")
        self.status_label.setStyleSheet("color: #28a745; font-weight: bold;")
        layout.addWidget(self.status_label)
        
        return header
    
    def create_employee_tab(self):
        """Create streamlined employee management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Form section
        form_group = QGroupBox("Add New Employee")
        form_layout = QVBoxLayout(form_group)
        
        # Create form fields
        self.emp_id = FormHelper.create_form_row(form_layout, "Employee ID", QLineEdit(), True)
        self.emp_name = FormHelper.create_form_row(form_layout, "Name", QLineEdit(), True)
        self.emp_skills = FormHelper.create_form_row(form_layout, "Skills (comma separated)", QLineEdit())
        self.emp_experience = FormHelper.create_form_row(form_layout, "Experience (years)", QLineEdit())
        self.emp_certifications = FormHelper.create_form_row(form_layout, "Certifications", QLineEdit())
        self.emp_goals = FormHelper.create_form_row(form_layout, "Career Goals", QLineEdit())
        self.emp_availability = FormHelper.create_form_row(form_layout, "Availability Date", QLineEdit())
        self.emp_preferences = FormHelper.create_form_row(form_layout, "Preferences (JSON)", QTextEdit())
        
        self.emp_preferences.setMaximumHeight(60)
        self.emp_preferences.setPlaceholderText('{"location": "New York", "department": "Engineering"}')
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = ModernButton("Add Employee", "➕", "success")
        add_btn.clicked.connect(self.add_employee)
        clear_btn = ModernButton("Clear Form", "🗑️", "secondary")
        clear_btn.clicked.connect(self.clear_employee_form)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        form_layout.addLayout(btn_layout)
        
        # Employee list
        list_group = QGroupBox("Employee Database")
        list_layout = QVBoxLayout(list_group)
        
        self.employee_table = EnhancedTableWidget(["ID", "Name", "Skills", "Experience"])
        list_layout.addWidget(self.employee_table)
        
        refresh_btn = ModernButton("Refresh", "🔄", "secondary")
        refresh_btn.clicked.connect(self.refresh_employee_list)
        list_layout.addWidget(refresh_btn)
        
        # Add to tab
        layout.addWidget(form_group)
        layout.addWidget(list_group)
        self.tabs.addTab(tab, "👥 Employees")
    
    def create_role_tab(self):
        """Create streamlined role management tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Form section
        form_group = QGroupBox("Add New Role")
        form_layout = QVBoxLayout(form_group)
        
        # Create form fields in a grid for better space usage
        grid_layout = QGridLayout()
        
        self.role_id = QLineEdit()
        self.role_project = QLineEdit()
        self.role_title = QLineEdit()
        self.role_req_skills = QLineEdit()
        self.role_location = QLineEdit()
        self.role_start_date = QLineEdit()
        self.role_duration = QLineEdit()
        self.role_priority = QLineEdit()
        
        # Add to grid
        grid_layout.addWidget(QLabel("Role ID*:"), 0, 0)
        grid_layout.addWidget(self.role_id, 0, 1)
        grid_layout.addWidget(QLabel("Project ID:"), 0, 2)
        grid_layout.addWidget(self.role_project, 0, 3)
        
        grid_layout.addWidget(QLabel("Title*:"), 1, 0)
        grid_layout.addWidget(self.role_title, 1, 1)
        grid_layout.addWidget(QLabel("Location:"), 1, 2)
        grid_layout.addWidget(self.role_location, 1, 3)
        
        grid_layout.addWidget(QLabel("Required Skills:"), 2, 0)
        grid_layout.addWidget(self.role_req_skills, 2, 1, 1, 3)
        
        grid_layout.addWidget(QLabel("Start Date:"), 3, 0)
        grid_layout.addWidget(self.role_start_date, 3, 1)
        grid_layout.addWidget(QLabel("Duration (weeks):"), 3, 2)
        grid_layout.addWidget(self.role_duration, 3, 3)
        
        form_layout.addLayout(grid_layout)
        
        # Buttons
        btn_layout = QHBoxLayout()
        add_btn = ModernButton("Add Role", "➕", "warning")
        add_btn.clicked.connect(self.add_role)
        clear_btn = ModernButton("Clear Form", "🗑️", "secondary")
        clear_btn.clicked.connect(self.clear_role_form)
        
        btn_layout.addWidget(add_btn)
        btn_layout.addWidget(clear_btn)
        btn_layout.addStretch()
        form_layout.addLayout(btn_layout)
        
        # Role list
        list_group = QGroupBox("Role Database")
        list_layout = QVBoxLayout(list_group)
        
        self.role_table = EnhancedTableWidget(["ID", "Title", "Skills", "Location"])
        list_layout.addWidget(self.role_table)
        
        refresh_btn = ModernButton("Refresh", "🔄", "secondary")
        refresh_btn.clicked.connect(self.refresh_role_list)
        list_layout.addWidget(refresh_btn)
        
        layout.addWidget(form_group)
        layout.addWidget(list_group)
        self.tabs.addTab(tab, "📋 Roles")
    
    def create_matching_tab(self):
        """Create enhanced matching tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_group = QGroupBox("AI Matching Controls")
        controls_layout = QHBoxLayout(controls_group)
        
        # Match type
        controls_layout.addWidget(QLabel("Match Type:"))
        self.match_type = QComboBox()
        self.match_type.addItems(["Employee → Roles", "Role → Employees"])
        controls_layout.addWidget(self.match_type)
        
        # ID input
        controls_layout.addWidget(QLabel("ID:"))
        self.match_id = QLineEdit()
        self.match_id.setPlaceholderText("Enter employee or role ID")
        controls_layout.addWidget(self.match_id)
        
        # Buttons
        build_btn = ModernButton("Build Matrix", "⚙️", "secondary")
        build_btn.clicked.connect(self.build_matrix)
        match_btn = ModernButton("Find Matches", "🔍", "primary")
        match_btn.clicked.connect(self.find_matches)
        
        controls_layout.addWidget(build_btn)
        controls_layout.addWidget(match_btn)
        controls_layout.addStretch()
        
        # Results
        results_group = QGroupBox("Match Results")
        results_layout = QVBoxLayout(results_group)
        
        self.match_results = EnhancedTableWidget(["Match", "Score", "Details", "Compatibility"])
        results_layout.addWidget(self.match_results)
        
        layout.addWidget(controls_group)
        layout.addWidget(results_group)
        self.tabs.addTab(tab, "✨ Matching")
    
    def create_analysis_tab(self):
        """Create analysis tab"""
        tab = QWidget()
        layout = QVBoxLayout(tab)
        
        # Controls
        controls_group = QGroupBox("Skill Gap Analysis")
        controls_layout = QHBoxLayout(controls_group)
        
        controls_layout.addWidget(QLabel("Department:"))
        self.dept_filter = QLineEdit()
        self.dept_filter.setPlaceholderText("Leave blank for all departments")
        controls_layout.addWidget(self.dept_filter)
        
        analyze_btn = ModernButton("Analyze Gaps", "📊", "primary")
        analyze_btn.clicked.connect(self.analyze_gaps)
        controls_layout.addWidget(analyze_btn)
        controls_layout.addStretch()
        
        # Results
        results_group = QGroupBox("Analysis Results")
        results_layout = QVBoxLayout(results_group)
        
        self.gap_results = QTextEdit()
        self.gap_results.setReadOnly(True)
        self.gap_results.setFont(QFont("Courier", 10))
        results_layout.addWidget(self.gap_results)
        
        layout.addWidget(controls_group)
        layout.addWidget(results_group)
        self.tabs.addTab(tab, "📈 Analysis")
    
    def get_app_stylesheet(self) -> str:
        """Get modern application stylesheet"""
        return """
            QMainWindow {
                background-color: #f8f9fa;
            }
            QGroupBox {
                font-weight: bold;
                border: 2px solid #dee2e6;
                border-radius: 8px;
                margin-top: 1ex;
                padding-top: 10px;
                background: white;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 10px 0 10px;
                color: #495057;
            }
            QLineEdit, QTextEdit, QComboBox {
                border: 1px solid #ced4da;
                border-radius: 4px;
                padding: 6px;
                background: white;
                selection-background-color: #007bff;
            }
            QLineEdit:focus, QTextEdit:focus, QComboBox:focus {
                border-color: #007bff;
                box-shadow: 0 0 0 2px rgba(0,123,255,0.25);
            }
        """
    
    def get_tab_stylesheet(self) -> str:
        """Get tab widget stylesheet"""
        return """
            QTabWidget::pane {
                border: 1px solid #dee2e6;
                border-radius: 4px;
                background: white;
            }
            QTabBar::tab {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #e9ecef, stop:1 #ced4da);
                color: #495057;
                padding: 12px 20px;
                border: 1px solid #dee2e6;
                border-radius: 4px 4px 0 0;
                margin-right: 2px;
                font-weight: bold;
            }
            QTabBar::tab:selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #4a9fdc, stop:1 #3a7fc4);
                color: white;
            }
            QTabBar::tab:hover:!selected {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                                          stop:0 #f8f9fa, stop:1 #e9ecef);
            }
        """
    
    # Event handlers
    def add_employee(self):
        """Add employee with validation"""
        try:
            # Collect form data
            data = {
                'id': self.emp_id.text().strip(),
                'name': self.emp_name.text().strip(),
                'skills': self.emp_skills.text().strip(),
                'experience': self.emp_experience.text().strip(),
                'certifications': self.emp_certifications.text().strip(),
                'goals': self.emp_goals.text().strip(),
                'date': self.emp_availability.text().strip(),
                'preferences': self.emp_preferences.toPlainText().strip()
            }
            
            # Validate
            is_valid, errors = FormHelper.validate_form_data(data)
            if not is_valid:
                self.show_status("; ".join(errors), "error")
                return
            
            # Parse and create employee
            skills = [s.strip() for s in data['skills'].split(",") if s.strip()] if data['skills'] else []
            certs = [c.strip() for c in data['certifications'].split(",") if c.strip()] if data['certifications'] else []
            goals = [g.strip() for g in data['goals'].split(",") if g.strip()] if data['goals'] else []
            
            # Parse preferences JSON
            try:
                preferences = json.loads(data['preferences']) if data['preferences'] else {}
            except json.JSONDecodeError:
                self.show_status("Invalid JSON format in preferences", "error")
                return
            
            # Create employee
            employee = Employee(
                employee_id=data['id'],
                name=data['name'],
                skills=skills,
                experience_years=int(data['experience']) if data['experience'] else 0,
                certifications=certs,
                career_goals=goals,
                past_projects=[],
                availability_date=data['date'] if data['date'] else datetime.now().strftime("%Y-%m-%d"),
                preferences=preferences
            )
            
            # Add to engine
            if self.engine.add_employee(employee):
                self.show_status(f"Employee {data['name']} added successfully!", "success")
                self.clear_employee_form()
                self.refresh_employee_list()
            else:
                self.show_status(f"Employee {data['id']} already exists", "warning")
                
        except ValueError as e:
            self.show_status(f"Invalid input: {str(e)}", "error")
        except Exception as e:
            self.show_status(f"Error adding employee: {str(e)}", "error")
    
    def add_role(self):
        """Add role with validation"""
        try:
            # Collect form data
            data = {
                'id': self.role_id.text().strip(),
                'project': self.role_project.text().strip(),
                'title': self.role_title.text().strip(),
                'skills': self.role_req_skills.text().strip(),
                'location': self.role_location.text().strip(),
                'date': self.role_start_date.text().strip(),
                'duration': self.role_duration.text().strip(),
                'priority': self.role_priority.text().strip()
            }
            
            # Validate required fields
            if not data['id'] or not data['title']:
                self.show_status("Role ID and Title are required", "error")
                return
            
            # Parse skills
            req_skills = [s.strip() for s in data['skills'].split(",") if s.strip()] if data['skills'] else []
            
            # Create role
            role = ProjectRole(
                role_id=data['id'],
                project_id=data['project'] if data['project'] else f"proj_{data['id']}",
                title=data['title'],
                required_skills=req_skills,
                nice_to_have_skills=[],
                required_certifications=[],
                start_date=data['date'] if data['date'] else datetime.now().strftime("%Y-%m-%d"),
                duration_weeks=int(data['duration']) if data['duration'] else 4,
                location=data['location'] if data['location'] else "Remote",
                priority=int(data['priority']) if data['priority'] else 3
            )
            
            # Add to engine
            if self.engine.add_role(role):
                self.show_status(f"Role {data['title']} added successfully!", "success")
                self.clear_role_form()
                self.refresh_role_list()
            else:
                self.show_status(f"Role {data['id']} already exists", "warning")
                
        except ValueError as e:
            self.show_status(f"Invalid input: {str(e)}", "error")
        except Exception as e:
            self.show_status(f"Error adding role: {str(e)}", "error")
    
    def clear_employee_form(self):
        """Clear employee form"""
        for widget in [self.emp_id, self.emp_name, self.emp_skills, self.emp_experience,
                      self.emp_certifications, self.emp_goals, self.emp_availability]:
            widget.clear()
        self.emp_preferences.clear()
    
    def clear_role_form(self):
        """Clear role form"""
        for widget in [self.role_id, self.role_project, self.role_title, self.role_req_skills,
                      self.role_location, self.role_start_date, self.role_duration, self.role_priority]:
            widget.clear()
    
    def refresh_employee_list(self):
        """Refresh employee table"""
        try:
            self.employee_table.setRowCount(len(self.engine.employees))
            
            for row, emp in enumerate(self.engine.employees):
                self.employee_table.setItem(row, 0, QTableWidgetItem(emp.employee_id))
                self.employee_table.setItem(row, 1, QTableWidgetItem(emp.name))
                self.employee_table.setItem(row, 2, QTableWidgetItem(", ".join(emp.skills[:3]) + "..." if len(emp.skills) > 3 else ", ".join(emp.skills)))
                self.employee_table.setItem(row, 3, QTableWidgetItem(str(emp.experience_years)))
                
            self.show_status(f"Employee list updated - {len(self.engine.employees)} employees", "info")
            
        except Exception as e:
            self.show_status(f"Error refreshing employee list: {str(e)}", "error")
    
    def refresh_role_list(self):
        """Refresh role table"""
        try:
            self.role_table.setRowCount(len(self.engine.roles))
            
            for row, role in enumerate(self.engine.roles):
                self.role_table.setItem(row, 0, QTableWidgetItem(role.role_id))
                self.role_table.setItem(row, 1, QTableWidgetItem(role.title))
                self.role_table.setItem(row, 2, QTableWidgetItem(", ".join(role.required_skills[:3]) + "..." if len(role.required_skills) > 3 else ", ".join(role.required_skills)))
                self.role_table.setItem(row, 3, QTableWidgetItem(role.location))
                
            self.show_status(f"Role list updated - {len(self.engine.roles)} roles", "info")
            
        except Exception as e:
            self.show_status(f"Error refreshing role list: {str(e)}", "error")
    
    def build_matrix(self):
        """Build skill matrix"""
        try:
            self.show_status("Building skill matrix...", "processing")
            QApplication.processEvents()  # Update UI
            
            self.engine.build_skill_matrix()
            self.show_status("Skill matrix built successfully!", "success")
            
        except Exception as e:
            self.show_status(f"Error building matrix: {str(e)}", "error")
    
    def find_matches(self):
        """Find matches based on selection"""
        try:
            match_type = self.match_type.currentText()
            entity_id = self.match_id.text().strip()
            
            if not entity_id:
                self.show_status("Please enter an ID to match", "warning")
                return
            
            self.show_status("Finding matches...", "processing")
            QApplication.processEvents()
            
            # Build matrix if not exists
            if self.engine.skill_matrix is None:
                self.engine.build_skill_matrix()
            
            # Find matches
            if "Employee" in match_type:
                results = self.engine.match_employee_to_roles(entity_id)
                self.display_role_matches(results)
            else:
                results = self.engine.match_role_to_employees(entity_id)
                self.display_employee_matches(results)
            
            if results:
                self.show_status(f"Found {len(results)} matches", "success")
            else:
                self.show_status("No matches found", "info")
                
        except Exception as e:
            self.show_status(f"Error finding matches: {str(e)}", "error")
    
    def display_role_matches(self, results):
        """Display role matching results"""
        self.match_results.setRowCount(len(results))
        
        for row, match in enumerate(results):
            role = match['role']
            
            # Create items with proper formatting
            name_item = QTableWidgetItem(role['title'])
            score_item = QTableWidgetItem(f"{match['score']:.2f}")
            score_item.setTextAlignment(Qt.AlignCenter)
            
            details_item = QTableWidgetItem(match['details'])
            
            # Color-coded compatibility
            compat_item = QTableWidgetItem(self.get_compatibility_text(match['score']))
            compat_item.setBackground(self.get_score_color(match['score']))
            compat_item.setTextAlignment(Qt.AlignCenter)
            
            # Set items
            self.match_results.setItem(row, 0, name_item)
            self.match_results.setItem(row, 1, score_item)
            self.match_results.setItem(row, 2, details_item)
            self.match_results.setItem(row, 3, compat_item)
    
    def display_employee_matches(self, results):
        """Display employee matching results"""
        self.match_results.setRowCount(len(results))
        
        for row, match in enumerate(results):
            emp = match['employee']
            
            # Create items
            name_item = QTableWidgetItem(emp['name'])
            score_item = QTableWidgetItem(f"{match['score']:.2f}")
            score_item.setTextAlignment(Qt.AlignCenter)
            
            details_item = QTableWidgetItem(match['details'])
            
            # Color-coded compatibility
            compat_item = QTableWidgetItem(self.get_compatibility_text(match['score']))
            compat_item.setBackground(self.get_score_color(match['score']))
            compat_item.setTextAlignment(Qt.AlignCenter)
            
            # Set items
            self.match_results.setItem(row, 0, name_item)
            self.match_results.setItem(row, 1, score_item)
            self.match_results.setItem(row, 2, details_item)
            self.match_results.setItem(row, 3, compat_item)
    
    def analyze_gaps(self):
        """Analyze skill gaps"""
        try:
            dept_filter = self.dept_filter.text().strip() or None
            
            self.show_status("Analyzing skill gaps...", "processing")
            QApplication.processEvents()
            
            gaps = self.engine.analyze_skill_gaps(dept_filter)
            
            if not gaps:
                self.gap_results.setText("✅ No significant skill gaps found!")
                self.show_status("No skill gaps found", "success")
                return
            
            # Create detailed report
            report = "📊 SKILL GAP ANALYSIS REPORT\n"
            report += "=" * 50 + "\n\n"
            
            if dept_filter:
                report += f"📍 Department: {dept_filter}\n\n"
            
            report += f"{'Skill':<25}{'Required':>8}{'Available':>10}{'Gap':>8}{'Priority':>10}\n"
            report += "-" * 65 + "\n"
            
            for skill, data in list(gaps.items())[:15]:  # Top 15 gaps
                priority = "🔴 HIGH" if data['gap'] > 3 else "🟡 MED" if data['gap'] > 1 else "🟢 LOW"
                report += f"{skill:<25}{data['required']:>8}{data['available']:>10}{data['gap']:>8}{priority:>10}\n"
            
            report += f"\n📈 Total skills analyzed: {len(gaps)}\n"
            report += f"🎯 Critical gaps (>3): {sum(1 for gap in gaps.values() if gap['gap'] > 3)}\n"
            
            self.gap_results.setText(report)
            self.show_status(f"Analysis complete - {len(gaps)} skill gaps found", "success")
            
        except Exception as e:
            self.show_status(f"Error analyzing gaps: {str(e)}", "error")
    
    def get_compatibility_text(self, score: float) -> str:
        """Get compatibility text based on score"""
        if score >= 0.8:
            return "🟢 Excellent"
        elif score >= 0.6:
            return "🟡 Good" 
        elif score >= 0.4:
            return "🟠 Fair"
        else:
            return "🔴 Poor"
    
    def get_score_color(self, score: float) -> QColor:
        """Get color based on match score"""
        if score >= 0.8:
            return QColor(220, 255, 220)  # Light green
        elif score >= 0.6:
            return QColor(255, 255, 200)  # Light yellow
        elif score >= 0.4:
            return QColor(255, 230, 200)  # Light orange
        else:
            return QColor(255, 220, 220)  # Light red
    
    def load_sample_data(self):
        """Load optimized sample data"""
        try:
            # Sample employees with realistic data
            sample_employees = [
                Employee("emp001", "John Doe", 
                        ["Python", "Machine Learning", "Data Analysis", "SQL", "TensorFlow"],
                        5, ["AWS Certified", "Google Cloud Professional"],
                        ["Become ML architect", "Lead AI projects"],
                        ["Project Alpha", "Project Beta"],
                        "2023-10-01",
                        {"location": "New York", "department": "AI"}),
                
                Employee("emp002", "Jane Smith",
                        ["Java", "Spring Boot", "Microservices", "AWS", "Kubernetes"],
                        3, ["AWS Solutions Architect"],
                        ["Cloud architecture", "Lead engineering teams"],
                        ["Project Gamma"],
                        "2023-09-15",
                        {"location": "Remote", "department": "Engineering"}),
                
                Employee("emp003", "Mike Johnson",
                        ["React", "Node.js", "JavaScript", "TypeScript", "GraphQL"],
                        4, ["AWS Developer Associate"],
                        ["Full-stack development", "Technical leadership"],
                        ["E-commerce Platform", "Mobile App"],
                        "2023-11-01",
                        {"location": "San Francisco", "department": "Frontend"})
            ]
            
            # Sample roles
            sample_roles = [
                ProjectRole("role001", "proj001", "Senior ML Engineer",
                           ["Python", "Machine Learning", "TensorFlow", "Data Analysis"],
                           ["PyTorch", "MLOps", "Docker"],
                           ["AWS Certified"],
                           "2023-10-15", 26, "New York", 1),
                
                ProjectRole("role002", "proj002", "Cloud Solutions Architect", 
                           ["AWS", "Microservices", "Java", "Kubernetes"],
                           ["Terraform", "Docker", "CI/CD"],
                           ["AWS Solutions Architect"],
                           "2023-11-01", 52, "Remote", 2),
                
                ProjectRole("role003", "proj003", "Full-Stack Developer",
                           ["React", "Node.js", "JavaScript", "GraphQL"],
                           ["TypeScript", "MongoDB", "Redis"],
                           [],
                           "2023-12-01", 20, "San Francisco", 3)
            ]
            
            # Add to engine
            for emp in sample_employees:
                self.engine.add_employee(emp)
            
            for role in sample_roles:
                self.engine.add_role(role)
            
            # Refresh displays
            self.refresh_employee_list()
            self.refresh_role_list()
            
            self.show_status("Sample data loaded successfully! 📊", "success")
            
        except Exception as e:
            self.show_status(f"Error loading sample data: {str(e)}", "error")

def main():
    """Main application entry point"""
    app = QApplication(sys.argv)
    
    # Set application properties
    app.setApplicationName("AI Talent Matching System")
    app.setApplicationVersion("2.0")
    app.setOrganizationName("TalentTech Solutions")
    
    # Apply modern style
    app.setStyle('Fusion')
    
    # Set application font
    font = QFont("Segoe UI", 9)
    app.setFont(font)
    
    # Create and show main window
    try:
        window = MainWindow()
        window.show()
        
        # Show startup message
        window.show_status("🚀 AI Talent Matching System v2.0 - Ready!", "success")
        
        sys.exit(app.exec_())
        
    except Exception as e:
        print(f"Failed to start application: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
