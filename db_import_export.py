# db_import_export.py
# Database Import/Export Functions for Court Scheduling System
# SQLite Version

import json
import pandas as pd
from datetime import datetime
from typing import List, Dict
from sqlalchemy.orm import Session
from database_models import (
    Case, Judge, Courtroom, Lawyer, Specialization, Schedule,
    DatabaseManager, DatabaseOperations)

class DataImporter:
    def __init__(self, session: Session):
        self.session = session
        self.db_ops = DatabaseOperations(session)

    def import_from_json(self, json_file: str) -> Dict[str, int]:
        """Import data from JSON file"""
        with open(json_file, 'r') as f:
            data = json.load(f)
        
        counts = {'cases': 0, 'judges': 0}
        
        # Simple import logic (expand as needed)
        if 'judges' in data:
            for j_data in data['judges']:
                # Check if exists
                if not self.session.query(Judge).filter_by(judge_id=j_data.get('judge_id')).first():
                    self.db_ops.create_judge(j_data)
                    counts['judges'] += 1
                    
        if 'cases' in data:
            for c_data in data['cases']:
                # Fix dates
                if 'filing_date' in c_data and c_data['filing_date']:
                    c_data['filing_date'] = datetime.fromisoformat(c_data['filing_date'])
                
                if not self.session.query(Case).filter_by(case_number=c_data.get('case_number')).first():
                    self.db_ops.create_case(c_data)
                    counts['cases'] += 1
        
        return counts

class DataExporter:
    def __init__(self, session: Session):
        self.session = session

    def export_to_json(self, output_file: str):
        data = {
            'timestamp': datetime.utcnow().isoformat(),
            'cases': [c.to_dict() for c in self.session.query(Case).all()],
            'judges': [j.to_dict() for j in self.session.query(Judge).all()],
            'courtrooms': [c.to_dict() for c in self.session.query(Courtroom).all()]
        }
        with open(output_file, 'w') as f:
            json.dump(data, f, indent=2, default=str)

class SampleDataGenerator:
    def __init__(self, session: Session):
        self.session = session
        self.db_ops = DatabaseOperations(session)

    def generate_complete_dataset(self, num_cases=50, num_judges=10, num_courtrooms=15, num_lawyers=30):
        import random
        
        # Judges
        for i in range(num_judges):
            judge_id = f"JDG_{i+1:03d}"
            # Check if judge exists
            if not self.session.query(Judge).filter_by(judge_id=judge_id).first():
                judge_data = {
                    'judge_id': judge_id,
                    'name': f"Judge {chr(65+i)}",
                    'experience_years': random.randint(5, 30),
                    'max_cases_per_day': 10,
                    'is_active': True
                }
                judge = self.db_ops.create_judge(judge_data)
                # Add specialization
                spec = self.session.query(Specialization).first()
                if spec and judge:
                    judge.specializations.append(spec)

        # Courtrooms
        for i in range(num_courtrooms):
            room_id = f"CR_{i+1:03d}"
            if not self.session.query(Courtroom).filter_by(courtroom_id=room_id).first():
                room_data = {
                    'courtroom_id': room_id,
                    'courtroom_number': f"Court-{i+1}",
                    'capacity': random.choice([50, 75, 100]),
                    'facilities': {'projector': True, 'ac': True},
                    'is_active': True
                }
                self.db_ops.create_courtroom(room_data)

        # Lawyers
        lawyers = self.session.query(Lawyer).all()
        if not lawyers:
            for i in range(num_lawyers):
                lawyer = Lawyer(
                    lawyer_id=f"LAW_{i+1:03d}",
                    name=f"Advocate {i+1}",
                    enrollment_number=f"EN{random.randint(10000,99999)}"
                )
                self.session.add(lawyer)
                lawyers.append(lawyer)
            self.session.commit()

        # Cases
        existing_cases = self.session.query(Case).count()
        if existing_cases < num_cases:
            case_types = ['Civil', 'Criminal', 'Family']
            for i in range(num_cases):
                case_type = random.choice(case_types)
                case_data = {
                    'case_number': f"{case_type[:3].upper()}/2024/{i+1+existing_cases:05d}",
                    'case_type': case_type,
                    'priority': random.randint(1, 3),
                    'estimated_duration': 60,
                    'filing_date': datetime.now(),
                    'required_specialization': f"{case_type} Law"
                }
                case = Case(**case_data)
                if len(lawyers) >= 2:
                    case.plaintiff_lawyers.append(random.choice(lawyers))
                    case.defendant_lawyers.append(random.choice(lawyers))
                self.session.add(case)
            
            self.session.commit()
            
        return {'cases': num_cases, 'judges': num_judges}

if __name__ == "__main__":
    DATABASE_URL = "sqlite:///court_scheduler.db"
    db_manager = DatabaseManager(DATABASE_URL)
    session = db_manager.get_session()
    
    generator = SampleDataGenerator(session)
    stats = generator.generate_complete_dataset()
    print(f"Sample data generated: {stats}")
    session.close()