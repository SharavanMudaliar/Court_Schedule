# Court Case Scheduling System Using Genetic Algorithm
# Complete Implementation
# Author: MCA Research Project
# Year: 2025-26

import random
import numpy as np
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import json
from dataclasses import dataclass, asdict
from collections import defaultdict
import copy

# =====================================================================
# DATA MODELS
# =====================================================================

@dataclass
class Case:
    """Represents a court case"""
    case_id: str
    case_type: str  # 'Civil', 'Criminal', 'Family'
    priority: int  # 1=Urgent, 2=Normal, 3=Low
    estimated_duration: int  # minutes
    filing_date: datetime
    last_hearing_date: Optional[datetime]
    plaintiff_lawyers: List[str]
    defendant_lawyers: List[str]
    required_specialization: str
    
    def __hash__(self):
        return hash(self.case_id)

@dataclass
class Judge:
    """Represents a judge"""
    judge_id: str
    name: str
    specialization: List[str]
    experience_years: int
    max_cases_per_day: int = 10
    
    def __hash__(self):
        return hash(self.judge_id)

@dataclass
class Courtroom:
    """Represents a courtroom"""
    courtroom_id: str
    courtroom_number: str
    capacity: int
    facilities: List[str]
    
    def __hash__(self):
        return hash(self.courtroom_id)

@dataclass
class Lawyer:
    """Represents a lawyer"""
    lawyer_id: str
    name: str
    enrollment_number: str
    
    def __hash__(self):
        return hash(self.lawyer_id)

@dataclass
class TimeSlot:
    """Represents a time slot"""
    date: datetime
    start_time: str  # "09:00"
    end_time: str    # "11:00"
    
    def __hash__(self):
        return hash((self.date.date(), self.start_time))
    
    def __eq__(self, other):
        return (self.date.date() == other.date.date() and 
                self.start_time == other.start_time)

@dataclass
class ScheduleGene:
    """Single gene in the chromosome - one case assignment"""
    case: Case
    judge: Judge
    courtroom: Courtroom
    time_slot: TimeSlot
    
    def __str__(self):
        return f"Case {self.case.case_id} | Judge {self.judge.judge_id} | Room {self.courtroom.courtroom_id} | {self.time_slot.date.date()} {self.time_slot.start_time}"

# =====================================================================
# GENETIC ALGORITHM CONFIGURATION
# =====================================================================

class GAConfig:
    """GA parameters configuration"""
    POPULATION_SIZE = 100
    MAX_GENERATIONS = 500
    CROSSOVER_RATE = 0.8
    MUTATION_RATE = 0.15
    TOURNAMENT_SIZE = 5
    ELITISM_RATE = 0.05
    STAGNATION_LIMIT = 50
    
    # Penalty weights
    PENALTY_JUDGE_CONFLICT = 1000
    PENALTY_COURTROOM_CONFLICT = 1000
    PENALTY_LAWYER_CONFLICT = 800
    PENALTY_WAITING_TIME = 10
    PENALTY_RESOURCE_UNDERUTILIZATION = 5
    PENALTY_PRIORITY_VIOLATION = 20
    PENALTY_SPECIALIZATION_MISMATCH = 500

# =====================================================================
# DATA GENERATOR (for testing)
# =====================================================================

class DataGenerator:
    """Generate sample data for testing"""
    
    @staticmethod
    def generate_cases(num_cases: int = 50) -> List[Case]:
        """Generate random cases"""
        cases = []
        case_types = ['Civil', 'Criminal', 'Family']
        specializations = ['Civil Law', 'Criminal Law', 'Family Law']
        
        for i in range(num_cases):
            case_type = random.choice(case_types)
            cases.append(Case(
                case_id=f"CASE_{i+1:04d}",
                case_type=case_type,
                priority=random.randint(1, 3),
                estimated_duration=random.choice([60, 90, 120, 180]),
                filing_date=datetime.now() - timedelta(days=random.randint(30, 365)),
                last_hearing_date=datetime.now() - timedelta(days=random.randint(1, 30)),
                plaintiff_lawyers=[f"LAW_{random.randint(1, 20):03d}"],
                defendant_lawyers=[f"LAW_{random.randint(1, 20):03d}"],
                required_specialization=case_type.replace(' ', '') + ' Law'
            ))
        return cases
    
    @staticmethod
    def generate_judges(num_judges: int = 10) -> List[Judge]:
        """Generate random judges"""
        judges = []
        specializations = [
            ['Civil Law'],
            ['Criminal Law'],
            ['Family Law'],
            ['Civil Law', 'Family Law'],
            ['Criminal Law', 'Civil Law']
        ]
        
        for i in range(num_judges):
            judges.append(Judge(
                judge_id=f"JDG_{i+1:03d}",
                name=f"Judge {chr(65+i)}",
                specialization=random.choice(specializations),
                experience_years=random.randint(5, 30),
                max_cases_per_day=random.randint(8, 12)
            ))
        return judges
    
    @staticmethod
    def generate_courtrooms(num_rooms: int = 15) -> List[Courtroom]:
        """Generate courtrooms"""
        courtrooms = []
        for i in range(num_rooms):
            courtrooms.append(Courtroom(
                courtroom_id=f"CR_{i+1:03d}",
                courtroom_number=f"Court-{i+1}",
                capacity=random.choice([50, 75, 100]),
                facilities=['Projector', 'AC', 'Recording']
            ))
        return courtrooms
    
    @staticmethod
    def generate_time_slots(start_date: datetime, num_days: int = 20) -> List[TimeSlot]:
        """Generate available time slots"""
        slots = []
        time_periods = [
            ("09:00", "11:00"),
            ("11:30", "13:30"),
            ("14:00", "16:00"),
            ("16:30", "18:30")
        ]
        
        current_date = start_date
        for day in range(num_days):
            # Skip weekends
            if current_date.weekday() < 5:  # Monday to Friday
                for start, end in time_periods:
                    slots.append(TimeSlot(
                        date=current_date,
                        start_time=start,
                        end_time=end
                    ))
            current_date += timedelta(days=1)
        
        return slots

# =====================================================================
# CHROMOSOME (Schedule Solution)
# =====================================================================

class Chromosome:
    """Represents a complete schedule solution"""
    
    def __init__(self, genes: List[ScheduleGene]):
        self.genes = genes
        self.fitness = 0.0
        self.constraint_violations = {}
    
    def __len__(self):
        return len(self.genes)
    
    def copy(self):
        """Deep copy of chromosome"""
        return Chromosome([
            ScheduleGene(
                case=gene.case,
                judge=gene.judge,
                courtroom=gene.courtroom,
                time_slot=TimeSlot(
                    date=gene.time_slot.date,
                    start_time=gene.time_slot.start_time,
                    end_time=gene.time_slot.end_time
                )
            ) for gene in self.genes
        ])
    
    def to_dict(self) -> List[Dict]:
        """Convert to dictionary for JSON serialization"""
        return [{
            'case_id': gene.case.case_id,
            'judge_id': gene.judge.judge_id,
            'courtroom_id': gene.courtroom.courtroom_id,
            'date': gene.time_slot.date.strftime('%Y-%m-%d'),
            'start_time': gene.time_slot.start_time,
            'end_time': gene.time_slot.end_time
        } for gene in self.genes]

# =====================================================================
# FITNESS EVALUATOR
# =====================================================================

class FitnessEvaluator:
    """Evaluates fitness of a schedule"""
    
    def __init__(self, config: GAConfig):
        self.config = config
    
    def evaluate(self, chromosome: Chromosome) -> float:
        """Calculate fitness score"""
        penalty = 0
        violations = defaultdict(int)
        
        # Check hard constraints
        penalty += self._check_judge_conflicts(chromosome, violations)
        penalty += self._check_courtroom_conflicts(chromosome, violations)
        penalty += self._check_lawyer_conflicts(chromosome, violations)
        penalty += self._check_specialization_match(chromosome, violations)
        
        # Check soft constraints
        penalty += self._calculate_waiting_time_penalty(chromosome, violations)
        penalty += self._calculate_resource_utilization_penalty(chromosome, violations)
        penalty += self._check_priority_violations(chromosome, violations)
        
        # Store violations for analysis
        chromosome.constraint_violations = dict(violations)
        
        # Fitness is inverse of penalty
        fitness = 1.0 / (1.0 + penalty)
        chromosome.fitness = fitness
        
        return fitness
    
    def _check_judge_conflicts(self, chromosome: Chromosome, violations: dict) -> float:
        """Check if judge has multiple cases at same time"""
        penalty = 0
        judge_schedule = defaultdict(list)
        
        for gene in chromosome.genes:
            key = (gene.judge.judge_id, gene.time_slot)
            judge_schedule[key].append(gene)
        
        for key, assignments in judge_schedule.items():
            if len(assignments) > 1:
                conflicts = len(assignments) - 1
                penalty += conflicts * self.config.PENALTY_JUDGE_CONFLICT
                violations['judge_conflicts'] += conflicts
        
        return penalty
    
    def _check_courtroom_conflicts(self, chromosome: Chromosome, violations: dict) -> float:
        """Check if courtroom has multiple cases at same time"""
        penalty = 0
        room_schedule = defaultdict(list)
        
        for gene in chromosome.genes:
            key = (gene.courtroom.courtroom_id, gene.time_slot)
            room_schedule[key].append(gene)
        
        for key, assignments in room_schedule.items():
            if len(assignments) > 1:
                conflicts = len(assignments) - 1
                penalty += conflicts * self.config.PENALTY_COURTROOM_CONFLICT
                violations['courtroom_conflicts'] += conflicts
        
        return penalty
    
    def _check_lawyer_conflicts(self, chromosome: Chromosome, violations: dict) -> float:
        """Check if lawyer has multiple cases at same time"""
        penalty = 0
        lawyer_schedule = defaultdict(list)
        
        for gene in chromosome.genes:
            all_lawyers = gene.case.plaintiff_lawyers + gene.case.defendant_lawyers
            for lawyer_id in all_lawyers:
                key = (lawyer_id, gene.time_slot)
                lawyer_schedule[key].append(gene)
        
        for key, assignments in lawyer_schedule.items():
            if len(assignments) > 1:
                conflicts = len(assignments) - 1
                penalty += conflicts * self.config.PENALTY_LAWYER_CONFLICT
                violations['lawyer_conflicts'] += conflicts
        
        return penalty
    
    def _check_specialization_match(self, chromosome: Chromosome, violations: dict) -> float:
        """Check if judge specialization matches case requirement"""
        penalty = 0
        
        for gene in chromosome.genes:
            if gene.case.required_specialization not in gene.judge.specialization:
                penalty += self.config.PENALTY_SPECIALIZATION_MISMATCH
                violations['specialization_mismatch'] += 1
        
        return penalty
    
    def _calculate_waiting_time_penalty(self, chromosome: Chromosome, violations: dict) -> float:
        """Calculate penalty for long waiting times"""
        penalty = 0
        
        for gene in chromosome.genes:
            if gene.case.last_hearing_date:
                days_waiting = (gene.time_slot.date - gene.case.last_hearing_date).days
                if days_waiting > 30:  # More than 30 days
                    penalty += (days_waiting - 30) * self.config.PENALTY_WAITING_TIME
                    violations['long_waiting_times'] += 1
        
        return penalty
    
    def _calculate_resource_utilization_penalty(self, chromosome: Chromosome, violations: dict) -> float:
        """Penalize poor resource utilization"""
        penalty = 0
        
        # Count unique judges and courtrooms used
        judges_used = set(gene.judge.judge_id for gene in chromosome.genes)
        rooms_used = set(gene.courtroom.courtroom_id for gene in chromosome.genes)
        
        # Encourage better distribution (simplified)
        if len(judges_used) < len(chromosome.genes) / 10:
            penalty += self.config.PENALTY_RESOURCE_UNDERUTILIZATION
            violations['underutilized_resources'] += 1
        
        return penalty
    
    def _check_priority_violations(self, chromosome: Chromosome, violations: dict) -> float:
        """Check if high-priority cases are scheduled early"""
        penalty = 0
        
        # Sort genes by date
        sorted_genes = sorted(chromosome.genes, key=lambda g: g.time_slot.date)
        
        for i, gene in enumerate(sorted_genes):
            if gene.case.priority == 1:  # Urgent
                # Penalty if urgent case is scheduled late
                if i > len(sorted_genes) * 0.3:  # Should be in first 30%
                    penalty += self.config.PENALTY_PRIORITY_VIOLATION
                    violations['priority_violations'] += 1
        
        return penalty

# =====================================================================
# GENETIC OPERATORS
# =====================================================================

class GeneticOperators:
    """Implements GA operators"""
    
    @staticmethod
    def initialize_population(cases: List[Case], judges: List[Judge], 
                            courtrooms: List[Courtroom], time_slots: List[TimeSlot],
                            pop_size: int) -> List[Chromosome]:
        """Create initial population"""
        population = []
        
        for _ in range(pop_size):
            genes = []
            for case in cases:
                gene = ScheduleGene(
                    case=case,
                    judge=random.choice(judges),
                    courtroom=random.choice(courtrooms),
                    time_slot=random.choice(time_slots)
                )
                genes.append(gene)
            
            population.append(Chromosome(genes))
        
        return population
    
    @staticmethod
    def tournament_selection(population: List[Chromosome], fitness_scores: List[float],
                            tournament_size: int, num_parents: int) -> List[Chromosome]:
        """Select parents using tournament selection"""
        parents = []
        
        for _ in range(num_parents):
            # Random tournament
            tournament_indices = random.sample(range(len(population)), tournament_size)
            tournament_fitness = [fitness_scores[i] for i in tournament_indices]
            
            # Winner
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            parents.append(population[winner_idx].copy())
        
        return parents
    
    @staticmethod
    def two_point_crossover(parent1: Chromosome, parent2: Chromosome) -> Tuple[Chromosome, Chromosome]:
        """Perform two-point crossover"""
        if len(parent1) < 3:
            return parent1.copy(), parent2.copy()
        
        point1 = random.randint(1, len(parent1) - 2)
        point2 = random.randint(point1 + 1, len(parent1))
        
        child1_genes = (parent1.genes[:point1] + 
                        parent2.genes[point1:point2] + 
                        parent1.genes[point2:])
        
        child2_genes = (parent2.genes[:point1] + 
                        parent1.genes[point1:point2] + 
                        parent2.genes[point2:])
        
        return Chromosome(child1_genes), Chromosome(child2_genes)
    
    @staticmethod
    def swap_mutation(chromosome: Chromosome, judges: List[Judge], 
                      courtrooms: List[Courtroom], time_slots: List[TimeSlot]) -> Chromosome:
        """Mutate by swapping or changing assignments"""
        mutated = chromosome.copy()
        
        if len(mutated.genes) < 2:
            return mutated
        
        mutation_type = random.choice(['swap', 'change_judge', 'change_room', 'change_time'])
        
        if mutation_type == 'swap':
            # Swap two random case assignments
            idx1, idx2 = random.sample(range(len(mutated.genes)), 2)
            mutated.genes[idx1], mutated.genes[idx2] = mutated.genes[idx2], mutated.genes[idx1]
        
        elif mutation_type == 'change_judge':
            # Change judge for random case
            idx = random.randint(0, len(mutated.genes) - 1)
            mutated.genes[idx].judge = random.choice(judges)
        
        elif mutation_type == 'change_room':
            # Change courtroom for random case
            idx = random.randint(0, len(mutated.genes) - 1)
            mutated.genes[idx].courtroom = random.choice(courtrooms)
        
        elif mutation_type == 'change_time':
            # Change time slot for random case
            idx = random.randint(0, len(mutated.genes) - 1)
            mutated.genes[idx].time_slot = random.choice(time_slots)
        
        return mutated

# =====================================================================
# MAIN GENETIC ALGORITHM
# =====================================================================

class CourtSchedulerGA:
    """Main Genetic Algorithm for Court Scheduling"""
    
    def __init__(self, cases: List[Case], judges: List[Judge], 
                 courtrooms: List[Courtroom], time_slots: List[TimeSlot],
                 config: GAConfig = None):
        self.cases = cases
        self.judges = judges
        self.courtrooms = courtrooms
        self.time_slots = time_slots
        self.config = config or GAConfig()
        self.evaluator = FitnessEvaluator(self.config)
        self.operators = GeneticOperators()
        
        self.best_solution = None
        self.best_fitness = float('-inf')
        self.fitness_history = []
        self.avg_fitness_history = []
    
    def run(self) -> Tuple[Chromosome, float]:
        """Execute the genetic algorithm"""
        print("=" * 80)
        print("COURT CASE SCHEDULING SYSTEM - GENETIC ALGORITHM")
        print("=" * 80)
        print(f"Cases: {len(self.cases)}")
        print(f"Judges: {len(self.judges)}")
        print(f"Population Size: {self.config.POPULATION_SIZE}")
        print("=" * 80)
        
        # Initialize population
        population = self.operators.initialize_population(
            self.cases, self.judges, self.courtrooms, self.time_slots,
            self.config.POPULATION_SIZE
        )
        
        stagnation_counter = 0
        generation = 0
        
        print("Starting evolution...\n")
        
        while generation < self.config.MAX_GENERATIONS:
            # Evaluate fitness
            fitness_scores = []
            generation_improved = False # Track improvement per generation
            
            for individual in population:
                fitness = self.evaluator.evaluate(individual)
                fitness_scores.append(fitness)
                
                if fitness > self.best_fitness:
                    self.best_fitness = fitness
                    self.best_solution = individual.copy()
                    generation_improved = True
            
            # Record statistics
            self.fitness_history.append(self.best_fitness)
            self.avg_fitness_history.append(np.mean(fitness_scores))
            
            # Stagnation check (FIXED LOGIC)
            if generation_improved:
                stagnation_counter = 0
            else:
                stagnation_counter += 1
            
            # Print progress
            if generation % 50 == 0 or generation == self.config.MAX_GENERATIONS - 1:
                print(f"Gen {generation:4d} | Best: {self.best_fitness:.6f} | Avg: {np.mean(fitness_scores):.6f}")
            
            # Check convergence
            if stagnation_counter > self.config.STAGNATION_LIMIT:
                print(f"\nConverged at generation {generation} (stagnation limit reached)")
                break
            
            # Selection
            num_parents = self.config.POPULATION_SIZE
            parents = self.operators.tournament_selection(
                population, fitness_scores, self.config.TOURNAMENT_SIZE, num_parents
            )
            
            # Create new population
            offspring = []
            
            # Elitism
            elite_count = max(1, int(self.config.POPULATION_SIZE * self.config.ELITISM_RATE))
            elite_indices = np.argsort(fitness_scores)[-elite_count:]
            offspring.extend([population[i].copy() for i in elite_indices])
            
            # Generate offspring
            while len(offspring) < self.config.POPULATION_SIZE:
                parent1, parent2 = random.sample(parents, 2)
                
                # Crossover
                if random.random() < self.config.CROSSOVER_RATE:
                    child1, child2 = self.operators.two_point_crossover(parent1, parent2)
                else:
                    child1, child2 = parent1.copy(), parent2.copy()
                
                # Mutation
                if random.random() < self.config.MUTATION_RATE:
                    child1 = self.operators.swap_mutation(child1, self.judges, self.courtrooms, self.time_slots)
                if random.random() < self.config.MUTATION_RATE:
                    child2 = self.operators.swap_mutation(child2, self.judges, self.courtrooms, self.time_slots)
                
                offspring.extend([child1, child2])
            
            # Replace population
            population = offspring[:self.config.POPULATION_SIZE]
            generation += 1
        
        print("\n" + "=" * 80)
        print("EVOLUTION COMPLETE")
        print(f"Final Generation: {generation}")
        print(f"Best Fitness: {self.best_fitness:.6f}")
        
        return self.best_solution, self.best_fitness

if __name__ == "__main__":
    # Simple test logic if run directly
    pass