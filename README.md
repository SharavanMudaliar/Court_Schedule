# ⚖️ Intelligent Court Case Scheduling System

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Prototype-green)

An AI-powered scheduling system designed to optimize court case allocation in the Indian Judicial System using Genetic Algorithms. This project automates the complex task of assigning cases to judges and courtrooms while minimizing conflicts and delays.

## 📖 Overview

The Indian judicial system faces a massive backlog of cases due to inefficient manual scheduling. This project solves that problem by using a **Genetic Algorithm (GA)** to mathematically find the optimal schedule.

It considers:
* **Hard Constraints:** Judge availability, Courtroom capacity, Lawyer conflicts.
* **Soft Constraints:** Prioritizing urgent cases, minimizing gap between hearings.

## 🌟 Features

- **🧬 Genetic Algorithm Engine:** Automatically evolves conflict-free schedules over 500+ generations.
- **📊 Real-time Dashboard:** A responsive React.js interface to visualize case distribution and GA progress.
- **⚡ Smart Constraints:**
  - **Priority Handling:** Urgent cases (Criminal/Family) are scheduled first.
  - **Conflict Detection:** Prevents double-booking of judges or courtrooms.
- **💾 Database Integrated:** Fully functional SQLite database with SQLAlchemy ORM.
- **📂 Data Management:** One-click bulk sample data generation and JSON export.

## 🛠️ Tech Stack

- **Backend:** Python 3.x, Flask (REST API)
- **AI/ML:** Custom Genetic Algorithm (Selection, Crossover, Mutation)
- **Frontend:** HTML5, React.js, Chart.js
- **Database:** SQLite (Default)
- **Libraries:** NumPy, Pandas, Flask-CORS

## 📂 Project Structure

```text
Court_Scheduler_Project/
│
├── app.py                  # The Web Server (Flask API Entry Point)
├── court_scheduler_ga.py   # The Core Genetic Algorithm Logic
├── database_models.py      # Database Schema (Cases, Judges, Schedules)
├── db_import_export.py     # Sample Data Generator & Export Tools
├── advanced_features.py    # Analytics & What-if Scenario Logic
├── dashboard.html          # The Frontend User Interface
├── requirements.txt        # List of Python dependencies
└── README.md               # Project Documentation
🚀 Installation & SetupPrerequisitesPython 3.10 or higher.pip (Python Package Manager).Step 1: Clone the RepositoryBashgit clone [https://github.com/YOUR_USERNAME/court-scheduler.git](https://github.com/YOUR_USERNAME/court-scheduler.git)
cd court-scheduler
Step 2: Install DependenciesBashpip install -r requirements.txt
Step 3: Initialize DatabaseRun this script to create the database file (court_scheduler.db) and tables:Bashpython database_models.py
# Output: ✓ Database tables created successfully
Step 4: Generate Sample DataPopulate the system with dummy Indian court cases and judges:Bashpython db_import_export.py
# Output: Sample data generated...
▶️ How to Run1. Start the Backend ServerOpen your terminal and run:Bashpython app.py
Keep this terminal window open! You should see a message like:* Running on http://0.0.0.0:50002. Open the DashboardNavigate to the project folder in your File Explorer.Double-click the dashboard.html file.It will open in your browser and connect to the running server automatically.🧪 Demo Script (How to Use)Dashboard Overview: Upon loading, check the top cards. You will see "Total Cases" and "Active Judges".Run AI: Click the blue "🧬 Run Genetic Algorithm" button.Watch: Look at your terminal window to see the generation logs scrolling.Result: Wait 10-20 seconds. The "Recent GA Runs" table at the bottom will update with the best fitness score.Analyze Charts: Observe the doughnut chart for case types and bar chart for priority distribution.Add Custom Case: Click "➕ Add Case" to manually input a high-priority case and see how the scheduler handles it.📡 API EndpointsMethodEndpointDescriptionGET/api/healthCheck system status.GET/api/casesFetch pending cases.POST/api/ga/runTrigger the Genetic Algorithm.GET/api/statistics/dashboardGet real-time stats.🤝 ContributingContributions are welcome! Please open an issue or submit a pull request.📜 LicenseThis project is licensed under the MIT License - see the LICENSE file for details.Developed for MCA Research Project (MCARP31), 2025.
### 2. File: `LICENSE`

```text
MIT License

Copyright (c) 2025 [Your Name]

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
3. File: .gitignorePlaintext# Python cache
__pycache__/
*.py[cod]
*$py.class

# Database
*.db
*.sqlite3

# Environment variables
.env
venv/
env/

# IDE settings
.vscode/
.idea/

# System files
.DS_Store
Thumbs.db
