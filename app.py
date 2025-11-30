# app.py
# Flask REST API for Court Scheduling System

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from functools import wraps
import traceback
from typing import Dict, Any

# Import database models and operations
from database_models import (
    DatabaseManager, DatabaseOperations, Case, Judge, Courtroom,
    Lawyer, Schedule, GenerationRun, FitnessHistory)
from db_import_export import DataImporter, DataExporter, SampleDataGenerator

app = Flask(__name__)
CORS(app)

app.config['SECRET_KEY'] = 'dev-secret-key'
app.config['DATABASE_URL'] = 'sqlite:///court_scheduler.db'

# Initialize database
db_manager = DatabaseManager(app.config['DATABASE_URL'])

# Create tables if they don't exist
with app.app_context():
    try:
        db_manager.create_tables()
    except Exception as e:
        print(f"Tables already exist or error: {e}")

def get_db_session():
    return db_manager.get_session()

def success_response(data: Any, message: str = "Success") -> Dict:
    return {'status': 'success', 'message': message, 'data': data}

def error_response(message: str, status_code: int = 400) -> tuple:
    return jsonify({'status': 'error', 'message': message}), status_code

def handle_errors(f):
    @wraps(f)
    def decorated_function(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(f"Error in {f.__name__}: {str(e)}")
            traceback.print_exc()
            return error_response(f"Internal server error: {str(e)}", 500)
    return decorated_function

# --- ENDPOINTS ---

@app.route('/api/health', methods=['GET'])
def health_check():
    return jsonify({'status': 'healthy', 'timestamp': datetime.utcnow().isoformat()})

@app.route('/api/cases', methods=['GET'])
@handle_errors
def get_cases():
    session = get_db_session()
    db_ops = DatabaseOperations(session)
    status = request.args.get('status')
    priority = request.args.get('priority', type=int)
    cases = db_ops.get_all_cases(status=status)
    if priority:
        cases = [c for c in cases if c.priority == priority]
    result = [case.to_dict() for case in cases]
    session.close()
    return jsonify(success_response(result, f"Retrieved {len(result)} cases"))

@app.route('/api/cases', methods=['POST'])
@handle_errors
def create_case():
    data = request.get_json()
    session = get_db_session()
    db_ops = DatabaseOperations(session)
    if 'filing_date' in data:
        data['filing_date'] = datetime.fromisoformat(data['filing_date'])
    else:
        data['filing_date'] = datetime.utcnow()
    case = db_ops.create_case(data)
    session.close()
    return jsonify(success_response(case.to_dict(), "Case created")), 201

@app.route('/api/schedules', methods=['GET'])
@handle_errors
def get_schedules():
    session = get_db_session()
    db_ops = DatabaseOperations(session)
    start = datetime.utcnow()
    end = start + timedelta(days=30)
    schedules = db_ops.get_schedules_by_date(start, end)
    result = [schedule.to_dict() for schedule in schedules]
    session.close()
    return jsonify(success_response(result))

@app.route('/api/statistics/dashboard', methods=['GET'])
@handle_errors
def get_dashboard_statistics():
    session = get_db_session()
    total_cases = session.query(Case).count()
    pending_cases = session.query(Case).filter(Case.status == 'Pending').count()
    scheduled_cases = session.query(Case).filter(Case.status == 'Scheduled').count()
    total_judges = session.query(Judge).filter(Judge.is_active == True).count()
    total_courtrooms = session.query(Courtroom).filter(Courtroom.is_active == True).count()
    
    today = datetime.utcnow()
    week_later = today + timedelta(days=7)
    upcoming_schedules = session.query(Schedule).filter(
        Schedule.scheduled_date >= today,
        Schedule.scheduled_date <= week_later
    ).count()
    
    stats = {
        'total_cases': total_cases,
        'pending_cases': pending_cases,
        'scheduled_cases': scheduled_cases,
        'completed_cases': 0,
        'total_judges': total_judges,
        'total_courtrooms': total_courtrooms,
        'upcoming_schedules': upcoming_schedules,
        'case_by_type': {},
        'case_by_priority': {}
    }
    
    for case_type in ['Civil', 'Criminal', 'Family']:
        count = session.query(Case).filter(Case.case_type == case_type).count()
        stats['case_by_type'][case_type] = count
    
    for priority in [1, 2, 3]:
        count = session.query(Case).filter(Case.priority == priority).count()
        stats['case_by_priority'][str(priority)] = count
        
    session.close()
    return jsonify(success_response(stats))

@app.route('/api/ga/run', methods=['POST'])
@handle_errors
def run_genetic_algorithm():
    data = request.get_json() or {}
    session = get_db_session()
    db_ops = DatabaseOperations(session)
    
    run_data = {
        'run_timestamp': datetime.utcnow(),
        'population_size': data.get('population_size', 100),
        'max_generations': data.get('max_generations', 500),
        'crossover_rate': 0.8,
        'mutation_rate': 0.15,
        'status': 'Running'
    }
    run = db_ops.create_generation_run(run_data)
    run_id = run.id
    session.close()
    
    return jsonify(success_response({'run_id': run_id, 'status': 'Running', 'message': 'GA started'}))

@app.route('/api/ga/runs', methods=['GET'])
@handle_errors
def get_ga_runs():
    session = get_db_session()
    db_ops = DatabaseOperations(session)
    runs = db_ops.get_recent_runs(limit=10)
    result = [run.to_dict() for run in runs]
    session.close()
    return jsonify(success_response(result))

@app.route('/api/data/generate-sample', methods=['POST'])
@handle_errors
def generate_sample_data():
    data = request.get_json() or {}
    session = get_db_session()
    generator = SampleDataGenerator(session)
    stats = generator.generate_complete_dataset(
        num_cases=data.get('num_cases', 50),
        num_judges=data.get('num_judges', 10),
        num_courtrooms=data.get('num_courtrooms', 15),
        num_lawyers=data.get('num_lawyers', 30)
    )
    session.close()
    return jsonify(success_response(stats, "Sample data generated"))

@app.route('/api/data/export', methods=['GET'])
@handle_errors
def export_data():
    session = get_db_session()
    exporter = DataExporter(session)
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as tmp:
        output_path = tmp.name
        exporter.export_to_json(output_path)
    session.close()
    
    with open(output_path, 'r') as f:
        data = f.read()
    os.unlink(output_path)
    
    from flask import Response
    return Response(data, mimetype='application/json', headers={'Content-Disposition': 'attachment;filename=court_data.json'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)