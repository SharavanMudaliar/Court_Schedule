# app.py
# Flask REST API for Court Scheduling System

import sys
import os
import random
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify
from flask_cors import CORS
from datetime import datetime, timedelta
from functools import wraps
import traceback
from typing import Dict, Any
from sqlalchemy import text 

# Import database models and operations
from database_models import (
    DatabaseManager, DatabaseOperations, Case, Judge, Courtroom,
    Lawyer, Schedule, GenerationRun, FitnessHistory, 
    case_plaintiff_lawyers, case_defendant_lawyers) 
from db_import_export import DataImporter, DataExporter, SampleDataGenerator

# Import the Genetic Algorithm logic
import court_scheduler_ga as ga

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
    return jsonify({'status': 'healthy', 'timestamp': datetime.now().isoformat()})

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
        data['filing_date'] = datetime.now()
    
    case = db_ops.create_case(data)
    case_data = case.to_dict() 
    session.close()
    return jsonify(success_response(case_data, "Case created")), 201

@app.route('/api/cases/<int:case_id>', methods=['DELETE'])
@handle_errors
def delete_case(case_id):
    session = get_db_session()
    db_ops = DatabaseOperations(session)
    success = db_ops.delete_case(case_id)
    session.close()
    
    if not success:
        return error_response(f"Case {case_id} not found", 404)
        
    return jsonify(success_response(None, "Case deleted successfully"))

@app.route('/api/data/reset', methods=['POST'])
@handle_errors
def reset_data():
    session = get_db_session()
    try:
        # CRITICAL FIX: Delete from association tables first
        session.execute(case_plaintiff_lawyers.delete())
        session.execute(case_defendant_lawyers.delete())
        
        session.query(Schedule).delete()
        session.query(Case).delete()
        session.query(FitnessHistory).delete()
        session.query(GenerationRun).delete()
        
        session.commit()
        return jsonify(success_response(None, "Database reset complete."))
    except Exception as e:
        session.rollback()
        raise e
    finally:
        session.close()

@app.route('/api/schedules', methods=['GET'])
@handle_errors
def get_schedules():
    session = get_db_session()
    schedules = session.query(Schedule).join(Case).join(Judge).join(Courtroom).order_by(Schedule.scheduled_date, Schedule.start_time).all()
    
    result = []
    for s in schedules:
        result.append({
            'id': s.id,
            'case_number': s.case.case_number,
            'case_priority': s.case.priority,
            'case_type': s.case.case_type,
            'judge_name': s.judge.name,
            'courtroom': s.courtroom.courtroom_number,
            'date': s.scheduled_date.strftime('%Y-%m-%d'),
            'time': f"{s.start_time} - {s.end_time}",
            'status': s.status
        })
        
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
    
    today = datetime.now()
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
    
    try:
        db_cases = session.query(Case).filter(Case.status == 'Pending').all()
        db_judges = session.query(Judge).filter(Judge.is_active == True).all()
        db_courtrooms = session.query(Courtroom).filter(Courtroom.is_active == True).all()
        
        if not db_cases:
            return error_response("No pending cases to schedule")

        ga_cases = [ga.Case(str(c.id), c.case_type, c.priority, c.estimated_duration, c.filing_date, c.last_hearing_date, [], [], c.required_specialization) for c in db_cases]
        ga_judges = [ga.Judge(str(j.id), j.name, [s.name for s in j.specializations], j.experience_years) for j in db_judges]
        ga_rooms = [ga.Courtroom(str(r.id), r.courtroom_number, r.capacity, []) for r in db_courtrooms]
        ga_slots = ga.DataGenerator.generate_time_slots(datetime.now(), num_days=5)
        
        scheduler = ga.CourtSchedulerGA(ga_cases, ga_judges, ga_rooms, ga_slots)
        best_solution, fitness = scheduler.run()
        
        for gene in best_solution.genes:
            new_schedule = Schedule(
                case_id=int(gene.case.case_id),
                judge_id=int(gene.judge.judge_id),
                courtroom_id=int(gene.courtroom.courtroom_id),
                scheduled_date=gene.time_slot.date,
                start_time=gene.time_slot.start_time,
                end_time=gene.time_slot.end_time,
                status='Scheduled',
                fitness_score=fitness
            )
            session.add(new_schedule)
            case_db = session.query(Case).get(int(gene.case.case_id))
            case_db.status = 'Scheduled'
            
        run_data = {
            'run_timestamp': datetime.now(),
            'population_size': 100,
            'max_generations': 500,
            'crossover_rate': 0.8,
            'mutation_rate': 0.15,
            'best_fitness': fitness,
            'status': 'Completed',
            'actual_generations': len(scheduler.fitness_history)
        }
        run = db_ops.create_generation_run(run_data)
        
        # FIX: Explicitly add history objects to session
        for i, fit_score in enumerate(scheduler.fitness_history):
            if i % 5 == 0 or i == len(scheduler.fitness_history) - 1:
                hist = FitnessHistory(
                    generation_run_id=run.id, 
                    generation_number=i,
                    best_fitness=float(fit_score),
                    avg_fitness=float(scheduler.avg_fitness_history[i]),
                    worst_fitness=0.0
                )
                session.add(hist)
        
        session.commit()
        return jsonify(success_response({'run_id': run.id, 'status': 'Completed', 'message': f'Scheduled {len(best_solution.genes)} cases successfully'}))
        
    except Exception as e:
        session.rollback()
        print(traceback.format_exc())
        return error_response(str(e))
    finally:
        session.close()

@app.route('/api/ga/history/<int:run_id>', methods=['GET'])
@handle_errors
def get_ga_history(run_id):
    session = get_db_session()
    if run_id == 0:
        latest_run = session.query(GenerationRun).order_by(GenerationRun.run_timestamp.desc()).first()
        if not latest_run:
            return jsonify(success_response([], "No runs found"))
        run_id = latest_run.id

    history = session.query(FitnessHistory).filter(FitnessHistory.generation_run_id == run_id).order_by(FitnessHistory.generation_number).all()
    result = [{ 'generation': h.generation_number, 'fitness': h.best_fitness, 'avg_fitness': h.avg_fitness } for h in history]
    
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
    format_type = request.args.get('format', 'json')
    session = get_db_session()
    exporter = DataExporter(session)
    import tempfile
    
    if format_type == 'excel':
        with tempfile.NamedTemporaryFile(mode='w+b', delete=False, suffix='.xlsx') as tmp:
            output_path = tmp.name
        exporter.export_to_excel(output_path)
        mimetype = 'application/vnd.openxmlformats-officedocument.spreadsheetml.sheet'
        filename = 'Court_Schedule_Report.xlsx'
        read_mode = 'rb'
    else:
        with tempfile.NamedTemporaryFile(mode='w+', delete=False, suffix='.json') as tmp:
            output_path = tmp.name
        exporter.export_to_json(output_path)
        mimetype = 'application/json'
        filename = 'court_data_backup.json'
        read_mode = 'r'
    
    session.close()
    with open(output_path, read_mode) as f:
        data = f.read()
    os.unlink(output_path)
    
    from flask import Response
    return Response(data, mimetype=mimetype, headers={'Content-Disposition': f'attachment;filename={filename}'})

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)