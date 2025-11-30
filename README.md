# ⚖️ Intelligent Court Case Scheduling System

![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)
![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![Status](https://img.shields.io/badge/Status-Prototype-green)

An AI-powered scheduling system designed to optimize court case allocation in the Indian Judicial System using Genetic Algorithms. This project automates the complex task of assigning cases to judges and courtrooms while minimizing conflicts and delays.

---

## 📖 Overview

The Indian judicial system faces a massive backlog of cases due to inefficient manual scheduling. This project solves that problem by using a **Genetic Algorithm (GA)** to mathematically find the optimal schedule.

It considers:

- **Hard Constraints:** Judge availability, Courtroom capacity, Lawyer conflicts  
- **Soft Constraints:** Prioritizing urgent cases, minimizing gap between hearings  

---

## 🌟 Features

- **🧬 Genetic Algorithm Engine:** Automatically evolves conflict-free schedules over 500+ generations  
- **📊 Real-time Dashboard:** A responsive interface to visualize case distribution and GA progress  
- **⚡ Smart Constraints:**
  - **Priority Handling:** Urgent cases (Criminal/Family) are scheduled first  
  - **Conflict Detection:** Prevents double-booking of judges or courtrooms  
- **💾 Database Integrated:** Fully functional SQLite database with SQLAlchemy ORM  
- **📂 Data Management:** One-click bulk sample data generation and JSON export  

---

## 🛠️ Tech Stack

- **Backend:** Python 3.x, Flask (REST API)  
- **AI/ML:** Custom Genetic Algorithm (Selection, Crossover, Mutation)  
- **Frontend:** HTML5, React.js, Chart.js  
- **Database:** SQLite (Default)  
- **Libraries:** NumPy, Pandas, Flask-CORS  

---

## 📂 Project Structure

```text
Court_Scheduler_Project/
│
├── app.py                  # The Web Server (Flask API Entry Point)
├── court_scheduler_ga.py   # The Core Genetic Algorithm Logic
├── database_models.py      # Database Schema (Cases, Judges, Schedules)
├── db_import_export.py     # Sample Data Generator & Export Tools
├── advanced_features.py   # Analytics & What-if Scenario Logic
├── dashboard.html          # The Frontend User Interface
├── requirements.txt       # List of Python dependencies
└── README.md               # Project Documentation
