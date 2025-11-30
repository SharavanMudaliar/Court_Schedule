# database_models.py
# SQLAlchemy ORM Models for Court Scheduling System
# SQLite Version

from sqlalchemy import create_engine, Column, Integer, String, DateTime, Boolean, Float, Text, ForeignKey, Table, JSON
from sqlalchemy.orm import declarative_base
from sqlalchemy.orm import relationship, sessionmaker, Session
from datetime import datetime
from typing import List, Optional
import json

Base = declarative_base()

# =====================================================================
# ASSOCIATION TABLES
# =====================================================================
case_plaintiff_lawyers = Table(
    'case_plaintiff_lawyers',
    Base.metadata,
    Column('case_id', Integer, ForeignKey('cases.id'), primary_key=True),
    Column('lawyer_id', Integer, ForeignKey('lawyers.id'), primary_key=True))

case_defendant_lawyers = Table(
    'case_defendant_lawyers',
    Base.metadata,
    Column('case_id', Integer, ForeignKey('cases.id'), primary_key=True),
    Column('lawyer_id', Integer, ForeignKey('lawyers.id'), primary_key=True))

judge_specializations = Table(
    'judge_specializations',
    Base.metadata,
    Column('judge_id', Integer, ForeignKey('judges.id'), primary_key=True),
    Column('specialization_id', Integer, ForeignKey('specializations.id'), primary_key=True))

# =====================================================================
# CORE ENTITIES
# =====================================================================
class Case(Base):
    __tablename__ = 'cases'
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_number = Column(String(50), unique=True, nullable=False, index=True)
    case_type = Column(String(50), nullable=False)
    priority = Column(Integer, nullable=False, default=2)
    estimated_duration = Column(Integer, nullable=False)
    filing_date = Column(DateTime, nullable=False)
    last_hearing_date = Column(DateTime, nullable=True)
    next_hearing_by = Column(DateTime, nullable=True)
    required_specialization = Column(String(100), nullable=False)
    status = Column(String(20), default='Pending')
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    plaintiff_lawyers = relationship('Lawyer', secondary=case_plaintiff_lawyers, backref='plaintiff_cases')
    defendant_lawyers = relationship('Lawyer', secondary=case_defendant_lawyers, backref='defendant_cases')
    schedules = relationship('Schedule', back_populates='case', cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'case_number': self.case_number,
            'case_type': self.case_type,
            'priority': self.priority,
            'estimated_duration': self.estimated_duration,
            'filing_date': self.filing_date.isoformat() if self.filing_date else None,
            'last_hearing_date': self.last_hearing_date.isoformat() if self.last_hearing_date else None,
            'required_specialization': self.required_specialization,
            'status': self.status,
            'plaintiff_lawyers': [l.lawyer_id for l in self.plaintiff_lawyers],
            'defendant_lawyers': [l.lawyer_id for l in self.defendant_lawyers]
        }

class Judge(Base):
    __tablename__ = 'judges'
    id = Column(Integer, primary_key=True, autoincrement=True)
    judge_id = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    experience_years = Column(Integer, nullable=False)
    max_cases_per_day = Column(Integer, default=10)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    specializations = relationship('Specialization', secondary=judge_specializations, backref='judges')
    availability = relationship('JudgeAvailability', back_populates='judge', cascade='all, delete-orphan')
    schedules = relationship('Schedule', back_populates='judge')

    def to_dict(self):
        return {
            'id': self.id,
            'judge_id': self.judge_id,
            'name': self.name,
            'experience_years': self.experience_years,
            'max_cases_per_day': self.max_cases_per_day,
            'is_active': self.is_active,
            'specializations': [s.name for s in self.specializations]
        }

class Courtroom(Base):
    __tablename__ = 'courtrooms'
    id = Column(Integer, primary_key=True, autoincrement=True)
    courtroom_id = Column(String(20), unique=True, nullable=False, index=True)
    courtroom_number = Column(String(20), nullable=False)
    capacity = Column(Integer, nullable=False)
    facilities = Column(JSON, nullable=True) # Changed to JSON for SQLite compatibility
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    schedules = relationship('Schedule', back_populates='courtroom')

    def to_dict(self):
        return {
            'id': self.id,
            'courtroom_id': self.courtroom_id,
            'courtroom_number': self.courtroom_number,
            'capacity': self.capacity,
            'facilities': self.facilities,
            'is_active': self.is_active
        }

class Lawyer(Base):
    __tablename__ = 'lawyers'
    id = Column(Integer, primary_key=True, autoincrement=True)
    lawyer_id = Column(String(20), unique=True, nullable=False, index=True)
    name = Column(String(100), nullable=False)
    enrollment_number = Column(String(50), unique=True, nullable=False)
    phone = Column(String(15), nullable=True)
    email = Column(String(100), nullable=True)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)

    def to_dict(self):
        return {
            'id': self.id,
            'lawyer_id': self.lawyer_id,
            'name': self.name,
            'enrollment_number': self.enrollment_number,
            'phone': self.phone,
            'email': self.email,
            'is_active': self.is_active
        }

class Specialization(Base):
    __tablename__ = 'specializations'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), unique=True, nullable=False)
    description = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

    def to_dict(self):
        return {'id': self.id, 'name': self.name, 'description': self.description}

class Schedule(Base):
    __tablename__ = 'schedules'
    id = Column(Integer, primary_key=True, autoincrement=True)
    case_id = Column(Integer, ForeignKey('cases.id'), nullable=False)
    judge_id = Column(Integer, ForeignKey('judges.id'), nullable=False)
    courtroom_id = Column(Integer, ForeignKey('courtrooms.id'), nullable=False)
    scheduled_date = Column(DateTime, nullable=False)
    start_time = Column(String(5), nullable=False)
    end_time = Column(String(5), nullable=False)
    status = Column(String(20), default='Scheduled')
    fitness_score = Column(Float, nullable=True)
    generation_run_id = Column(Integer, ForeignKey('generation_runs.id'), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    updated_at = Column(DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    case = relationship('Case', back_populates='schedules')
    judge = relationship('Judge', back_populates='schedules')
    courtroom = relationship('Courtroom', back_populates='schedules')
    generation_run = relationship('GenerationRun', back_populates='schedules')

    def to_dict(self):
        return {
            'id': self.id,
            'case_number': self.case.case_number if self.case else None,
            'judge_name': self.judge.name if self.judge else None,
            'courtroom_number': self.courtroom.courtroom_number if self.courtroom else None,
            'scheduled_date': self.scheduled_date.isoformat() if self.scheduled_date else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'status': self.status,
            'fitness_score': self.fitness_score
        }

class JudgeAvailability(Base):
    __tablename__ = 'judge_availability'
    id = Column(Integer, primary_key=True, autoincrement=True)
    judge_id = Column(Integer, ForeignKey('judges.id'), nullable=False)
    date = Column(DateTime, nullable=False)
    start_time = Column(String(5), nullable=False)
    end_time = Column(String(5), nullable=False)
    is_available = Column(Boolean, default=True)
    reason = Column(String(200), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    judge = relationship('Judge', back_populates='availability')

    def to_dict(self):
        return {
            'id': self.id,
            'judge_id': self.judge.judge_id,
            'date': self.date.isoformat() if self.date else None,
            'start_time': self.start_time,
            'end_time': self.end_time,
            'is_available': self.is_available,
            'reason': self.reason
        }

class GenerationRun(Base):
    __tablename__ = 'generation_runs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    run_timestamp = Column(DateTime, default=datetime.utcnow, nullable=False)
    population_size = Column(Integer, nullable=False)
    max_generations = Column(Integer, nullable=False)
    actual_generations = Column(Integer, nullable=True)
    crossover_rate = Column(Float, nullable=False)
    mutation_rate = Column(Float, nullable=False)
    best_fitness = Column(Float, nullable=True)
    avg_fitness = Column(Float, nullable=True)
    total_violations = Column(Integer, nullable=True)
    constraint_violations = Column(JSON, nullable=True)
    execution_time = Column(Float, nullable=True)
    convergence_generation = Column(Integer, nullable=True)
    status = Column(String(20), default='Running')
    notes = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    schedules = relationship('Schedule', back_populates='generation_run')
    fitness_history = relationship('FitnessHistory', back_populates='generation_run', cascade='all, delete-orphan')

    def to_dict(self):
        return {
            'id': self.id,
            'run_timestamp': self.run_timestamp.isoformat() if self.run_timestamp else None,
            'population_size': self.population_size,
            'max_generations': self.max_generations,
            'actual_generations': self.actual_generations,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'status': self.status
        }

class FitnessHistory(Base):
    __tablename__ = 'fitness_history'
    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_run_id = Column(Integer, ForeignKey('generation_runs.id'), nullable=False)
    generation_number = Column(Integer, nullable=False)
    best_fitness = Column(Float, nullable=False)
    avg_fitness = Column(Float, nullable=False)
    worst_fitness = Column(Float, nullable=False)
    violations = Column(Integer, nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    generation_run = relationship('GenerationRun', back_populates='fitness_history')

    def to_dict(self):
        return {
            'generation_number': self.generation_number,
            'best_fitness': self.best_fitness,
            'avg_fitness': self.avg_fitness,
            'worst_fitness': self.worst_fitness,
            'violations': self.violations
        }

class Scenario(Base):
    __tablename__ = 'scenarios'
    id = Column(Integer, primary_key=True, autoincrement=True)
    name = Column(String(100), nullable=False)
    description = Column(Text, nullable=True)
    configuration = Column(JSON, nullable=False)
    created_by = Column(String(100), nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    results = relationship('ScenarioResult', back_populates='scenario', cascade='all, delete-orphan')

class ScenarioResult(Base):
    __tablename__ = 'scenario_results'
    id = Column(Integer, primary_key=True, autoincrement=True)
    scenario_id = Column(Integer, ForeignKey('scenarios.id'), nullable=False)
    generation_run_id = Column(Integer, ForeignKey('generation_runs.id'), nullable=True)
    metrics = Column(JSON, nullable=False)
    analysis = Column(Text, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)
    scenario = relationship('Scenario', back_populates='results')

class ParetoFront(Base):
    __tablename__ = 'pareto_fronts'
    id = Column(Integer, primary_key=True, autoincrement=True)
    generation_run_id = Column(Integer, ForeignKey('generation_runs.id'), nullable=False)
    generation_number = Column(Integer, nullable=False)
    solution_data = Column(JSON, nullable=False)
    objective_values = Column(JSON, nullable=False)
    crowding_distance = Column(Float, nullable=True)
    rank = Column(Integer, nullable=True)
    created_at = Column(DateTime, default=datetime.utcnow)

class AuditLog(Base):
    __tablename__ = 'audit_logs'
    id = Column(Integer, primary_key=True, autoincrement=True)
    entity_type = Column(String(50), nullable=False)
    entity_id = Column(Integer, nullable=False)
    action = Column(String(20), nullable=False)
    old_values = Column(JSON, nullable=True)
    new_values = Column(JSON, nullable=True)
    user = Column(String(100), nullable=True)
    timestamp = Column(DateTime, default=datetime.utcnow)
    ip_address = Column(String(45), nullable=True)

class DatabaseManager:
    def __init__(self, database_url: str):
        self.engine = create_engine(database_url, echo=False)
        self.SessionLocal = sessionmaker(bind=self.engine, autocommit=False, autoflush=False)
    
    def create_tables(self):
        Base.metadata.create_all(self.engine)
        print("✓ Database tables created successfully")
    
    def get_session(self) -> Session:
        return self.SessionLocal()
    
    def close(self):
        self.engine.dispose()

class DatabaseOperations:
    def __init__(self, session: Session):
        self.session = session

    def create_case(self, case_data: dict) -> Case:
        case = Case(**case_data)
        self.session.add(case)
        self.session.commit()
        self.session.refresh(case)
        return case

    def get_case(self, case_id: int) -> Optional[Case]:
        return self.session.query(Case).filter(Case.id == case_id).first()

    def get_all_cases(self, status: Optional[str] = None) -> List[Case]:
        query = self.session.query(Case)
        if status:
            query = query.filter(Case.status == status)
        return query.all()

    def update_case(self, case_id: int, updates: dict) -> Case:
        case = self.get_case(case_id)
        if case:
            for key, value in updates.items():
                setattr(case, key, value)
            self.session.commit()
            self.session.refresh(case)
        return case

    def delete_case(self, case_id: int) -> bool:
        case = self.get_case(case_id)
        if case:
            self.session.delete(case)
            self.session.commit()
            return True
        return False

    def create_judge(self, judge_data: dict) -> Judge:
        judge = Judge(**judge_data)
        self.session.add(judge)
        self.session.commit()
        self.session.refresh(judge)
        return judge

    def get_judge(self, judge_id: int) -> Optional[Judge]:
        return self.session.query(Judge).filter(Judge.id == judge_id).first()

    def get_all_judges(self, active_only: bool = True) -> List[Judge]:
        query = self.session.query(Judge)
        if active_only:
            query = query.filter(Judge.is_active == True)
        return query.all()

    def create_courtroom(self, courtroom_data: dict) -> Courtroom:
        courtroom = Courtroom(**courtroom_data)
        self.session.add(courtroom)
        self.session.commit()
        self.session.refresh(courtroom)
        return courtroom

    def get_all_courtrooms(self, active_only: bool = True) -> List[Courtroom]:
        query = self.session.query(Courtroom)
        if active_only:
            query = query.filter(Courtroom.is_active == True)
        return query.all()

    def create_schedule(self, schedule_data: dict) -> Schedule:
        schedule = Schedule(**schedule_data)
        self.session.add(schedule)
        self.session.commit()
        self.session.refresh(schedule)
        return schedule

    def get_schedules_by_date(self, start_date: datetime, end_date: datetime) -> List[Schedule]:
        return self.session.query(Schedule).filter(
            Schedule.scheduled_date >= start_date,
            Schedule.scheduled_date <= end_date
        ).all()

    def create_generation_run(self, run_data: dict) -> GenerationRun:
        run = GenerationRun(**run_data)
        self.session.add(run)
        self.session.commit()
        self.session.refresh(run)
        return run

    def update_generation_run(self, run_id: int, updates: dict) -> GenerationRun:
        run = self.session.query(GenerationRun).filter(GenerationRun.id == run_id).first()
        if run:
            for key, value in updates.items():
                setattr(run, key, value)
            self.session.commit()
            self.session.refresh(run)
        return run

    def get_recent_runs(self, limit: int = 10) -> List[GenerationRun]:
        return self.session.query(GenerationRun).order_by(
            GenerationRun.run_timestamp.desc()
        ).limit(limit).all()

if __name__ == "__main__":
    DATABASE_URL = "sqlite:///court_scheduler.db"
    db_manager = DatabaseManager(DATABASE_URL)
    db_manager.create_tables()
    
    # Create default specializations
    session = db_manager.get_session()
    specs = ['Civil Law', 'Criminal Law', 'Family Law']
    for s in specs:
        if not session.query(Specialization).filter_by(name=s).first():
            session.add(Specialization(name=s))
    session.commit()
    session.close()
    
    print("✓ Database setup completed successfully!")
    db_manager.close()