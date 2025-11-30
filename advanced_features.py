# advanced_features.py
# Advanced Features for Court Scheduling System
# NSGA-II, Real-time Updates, What-if Analysis, Historical Analytics

import numpy as np
import random
from typing import List, Tuple, Dict, Optional
from datetime import datetime, timedelta
from dataclasses import dataclass
import json
from collections import defaultdict
import copy

# Import base classes
from database_models import (
    DatabaseManager, DatabaseOperations, Scenario, ScenarioResult,
    ParetoFront, GenerationRun, Session
)

# =====================================================================
# MULTI-OBJECTIVE OPTIMIZATION (NSGA-II)
# =====================================================================

@dataclass
class MultiObjectiveSolution:
    """Solution with multiple objectives"""
    genes: List  # Schedule genes
    objectives: List[float]  # [minimize_delay, minimize_conflicts, maximize_utilization]
    rank: int = 0
    crowding_distance: float = 0.0
    
    def dominates(self, other) -> bool:
        """Check if this solution dominates another (Pareto dominance)"""
        better_in_one = False
        for i in range(len(self.objectives)):
            if self.objectives[i] > other.objectives[i]:  # Assuming maximization
                return False
            if self.objectives[i] < other.objectives[i]:
                better_in_one = True
        return better_in_one

class NSGAII:
    """
    Non-dominated Sorting Genetic Algorithm II
    for multi-objective optimization of court scheduling
    """
    
    def __init__(self, population_size: int = 100, max_generations: int = 500):
        self.population_size = population_size
        self.max_generations = max_generations
        self.population: List[MultiObjectiveSolution] = []
        self.pareto_fronts: List[List[MultiObjectiveSolution]] = []
    
    def evaluate_objectives(self, solution: MultiObjectiveSolution) -> List[float]:
        """
        Evaluate multiple objectives for a solution
        
        Objectives to minimize:
        1. Total delay (waiting time for cases)
        2. Number of constraint violations (conflicts)
        3. Resource underutilization (negative of utilization)
        """
        # Objective 1: Minimize total delay
        total_delay = 0
        for gene in solution.genes:
            if gene.case.last_hearing_date:
                days_waiting = (gene.time_slot.date - gene.case.last_hearing_date).days
                total_delay += max(0, days_waiting - 30)  # Penalty after 30 days
        
        # Objective 2: Minimize conflicts
        conflicts = self._count_conflicts(solution)
        
        # Objective 3: Minimize underutilization (maximize utilization)
        judges_used = len(set(gene.judge.judge_id for gene in solution.genes))
        total_judges = 15  # Assume 15 total judges
        underutilization = total_judges - judges_used
        
        # Return objectives (all to be minimized)
        return [total_delay, conflicts, underutilization]
    
    def _count_conflicts(self, solution: MultiObjectiveSolution) -> int:
        """Count total conflicts in solution"""
        conflicts = 0
        
        # Judge conflicts
        judge_schedule = defaultdict(list)
        for gene in solution.genes:
            key = (gene.judge.judge_id, gene.time_slot)
            judge_schedule[key].append(gene)
        
        for assignments in judge_schedule.values():
            if len(assignments) > 1:
                conflicts += len(assignments) - 1
        
        # Courtroom conflicts
        room_schedule = defaultdict(list)
        for gene in solution.genes:
            key = (gene.courtroom.courtroom_id, gene.time_slot)
            room_schedule[key].append(gene)
        
        for assignments in room_schedule.values():
            if len(assignments) > 1:
                conflicts += len(assignments) - 1
        
        return conflicts
    
    def fast_non_dominated_sort(self, population: List[MultiObjectiveSolution]) -> List[List[MultiObjectiveSolution]]:
        """
        Fast non-dominated sorting algorithm
        Returns list of fronts (Pareto fronts)
        """
        fronts = [[]]
        
        for p in population:
            p.domination_count = 0
            p.dominated_solutions = []
            
            for q in population:
                if p.dominates(q):
                    p.dominated_solutions.append(q)
                elif q.dominates(p):
                    p.domination_count += 1
            
            if p.domination_count == 0:
                p.rank = 0
                fronts[0].append(p)
        
        i = 0
        while len(fronts[i]) > 0:
            next_front = []
            for p in fronts[i]:
                for q in p.dominated_solutions:
                    q.domination_count -= 1
                    if q.domination_count == 0:
                        q.rank = i + 1
                        next_front.append(q)
            i += 1
            if next_front:
                fronts.append(next_front)
        
        return fronts[:-1] if len(fronts) > 1 else fronts
    
    def calculate_crowding_distance(self, front: List[MultiObjectiveSolution]):
        """Calculate crowding distance for solutions in a front"""
        if len(front) <= 2:
            for sol in front:
                sol.crowding_distance = float('inf')
            return
        
        num_objectives = len(front[0].objectives)
        
        # Initialize distances
        for sol in front:
            sol.crowding_distance = 0
        
        # For each objective
        for obj_idx in range(num_objectives):
            # Sort by objective value
            front.sort(key=lambda x: x.objectives[obj_idx])
            
            # Boundary points get infinite distance
            front[0].crowding_distance = float('inf')
            front[-1].crowding_distance = float('inf')
            
            # Calculate distance for interior points
            obj_min = front[0].objectives[obj_idx]
            obj_max = front[-1].objectives[obj_idx]
            obj_range = obj_max - obj_min
            
            if obj_range == 0:
                continue
            
            for i in range(1, len(front) - 1):
                distance = (front[i+1].objectives[obj_idx] - front[i-1].objectives[obj_idx]) / obj_range
                front[i].crowding_distance += distance
    
    def selection(self, population: List[MultiObjectiveSolution], size: int) -> List[MultiObjectiveSolution]:
        """Tournament selection based on rank and crowding distance"""
        selected = []
        for _ in range(size):
            tournament = random.sample(population, min(2, len(population)))
            winner = self._compare_solutions(tournament[0], tournament[1])
            selected.append(copy.deepcopy(winner))
        return selected
    
    def _compare_solutions(self, sol1: MultiObjectiveSolution, sol2: MultiObjectiveSolution) -> MultiObjectiveSolution:
        """Compare two solutions based on rank and crowding distance"""
        if sol1.rank < sol2.rank:
            return sol1
        elif sol1.rank > sol2.rank:
            return sol2
        else:
            # Same rank, compare crowding distance
            return sol1 if sol1.crowding_distance > sol2.crowding_distance else sol2
    
    def run(self) -> List[List[MultiObjectiveSolution]]:
        """Execute NSGA-II algorithm"""
        print("=" * 80)
        print("NSGA-II MULTI-OBJECTIVE OPTIMIZATION")
        print("=" * 80)
        
        # Initialize population (placeholder - needs actual initialization)
        # In real implementation, integrate with CourtSchedulerGA
        
        for generation in range(self.max_generations):
            # Evaluate objectives for all solutions
            for solution in self.population:
                solution.objectives = self.evaluate_objectives(solution)
            
            # Non-dominated sorting
            fronts = self.fast_non_dominated_sort(self.population)
            
            # Calculate crowding distance for each front
            for front in fronts:
                self.calculate_crowding_distance(front)
            
            # Selection, crossover, mutation (placeholder)
            offspring = self.selection(self.population, self.population_size)
            
            # Combine parent and offspring
            combined = self.population + offspring
            
            # Sort and select next generation
            fronts = self.fast_non_dominated_sort(combined)
            next_population = []
            
            for front in fronts:
                if len(next_population) + len(front) <= self.population_size:
                    next_population.extend(front)
                else:
                    # Sort by crowding distance and take remaining
                    self.calculate_crowding_distance(front)
                    front.sort(key=lambda x: x.crowding_distance, reverse=True)
                    remaining = self.population_size - len(next_population)
                    next_population.extend(front[:remaining])
                    break
            
            self.population = next_population
            
            if generation % 50 == 0:
                print(f"Generation {generation}: {len(fronts[0])} solutions in Pareto front")
        
        # Return final Pareto fronts
        self.pareto_fronts = self.fast_non_dominated_sort(self.population)
        print(f"\nOptimization complete. {len(self.pareto_fronts[0])} Pareto-optimal solutions found.")
        return self.pareto_fronts

# =====================================================================
# REAL-TIME CONSTRAINT UPDATES
# =====================================================================

class RealTimeConstraintManager:
    """Manage real-time constraint updates"""
    
    def __init__(self, session: Session):
        self.session = session
        self.db_ops = DatabaseOperations(session)
        self.constraint_cache = {}
        self.update_listeners = []
    
    def update_judge_availability(self, judge_id: str, date: datetime, 
                                  start_time: str, end_time: str, 
                                  is_available: bool, reason: str = None):
        """Update judge availability in real-time"""
        from database_models import JudgeAvailability, Judge
        
        judge = self.session.query(Judge).filter(Judge.judge_id == judge_id).first()
        if not judge:
            raise ValueError(f"Judge {judge_id} not found")
        
        availability = JudgeAvailability(
            judge_id=judge.id,
            date=date,
            start_time=start_time,
            end_time=end_time,
            is_available=is_available,
            reason=reason
        )
        
        self.session.add(availability)
        self.session.commit()
        
        # Notify listeners
        self._notify_update('judge_availability', {
            'judge_id': judge_id,
            'date': date.isoformat(),
            'is_available': is_available
        })
        
        print(f"✓ Updated availability for {judge_id} on {date.date()}")
    
    def mark_courtroom_unavailable(self, courtroom_id: str, date: datetime,
                                   start_time: str, end_time: str, reason: str):
        """Mark courtroom as unavailable for maintenance, etc."""
        from database_models import Courtroom
        
        courtroom = self.session.query(Courtroom).filter(
            Courtroom.courtroom_id == courtroom_id
        ).first()
        
        if not courtroom:
            raise ValueError(f"Courtroom {courtroom_id} not found")
        
        # Store in constraint cache
        key = f"courtroom_{courtroom_id}_{date.date()}_{start_time}"
        self.constraint_cache[key] = {
            'type': 'courtroom_unavailable',
            'courtroom_id': courtroom_id,
            'date': date,
            'start_time': start_time,
            'end_time': end_time,
            'reason': reason
        }
        
        self._notify_update('courtroom_unavailable', self.constraint_cache[key])
        print(f"✓ Marked {courtroom_id} unavailable on {date.date()} {start_time}-{end_time}")
    
    def add_priority_case(self, case_id: int, urgent: bool = True):
        """Elevate case priority"""
        case = self.db_ops.get_case(case_id)
        if case:
            self.db_ops.update_case(case_id, {'priority': 1 if urgent else case.priority})
            self._notify_update('priority_change', {'case_id': case_id, 'priority': 1})
            print(f"✓ Updated priority for case {case_id}")
    
    def register_listener(self, callback):
        """Register a callback for constraint updates"""
        self.update_listeners.append(callback)
    
    def _notify_update(self, update_type: str, data: dict):
        """Notify all registered listeners"""
        for listener in self.update_listeners:
            try:
                listener(update_type, data)
            except Exception as e:
                print(f"Error notifying listener: {e}")
    
    def get_active_constraints(self, date: datetime) -> Dict:
        """Get all active constraints for a given date"""
        constraints = {
            'judge_unavailability': [],
            'courtroom_unavailability': [],
            'priority_cases': []
        }
        
        # Query judge availability
        from database_models import JudgeAvailability
        unavailable_judges = self.session.query(JudgeAvailability).filter(
            JudgeAvailability.date == date,
            JudgeAvailability.is_available == False
        ).all()
        
        for avail in unavailable_judges:
            constraints['judge_unavailability'].append(avail.to_dict())
        
        # Get priority cases
        from database_models import Case
        priority_cases = self.session.query(Case).filter(
            Case.priority == 1,
            Case.status == 'Pending'
        ).all()
        
        for case in priority_cases:
            constraints['priority_cases'].append(case.to_dict())
        
        return constraints

# =====================================================================
# WHAT-IF SCENARIO ANALYSIS
# =====================================================================

class WhatIfAnalyzer:
    """Perform what-if scenario analysis"""
    
    def __init__(self, session: Session):
        self.session = session
        self.db_ops = DatabaseOperations(session)
    
    def create_scenario(self, name: str, description: str, 
                       configuration: Dict) -> Scenario:
        """Create a new what-if scenario"""
        scenario = Scenario(
            name=name,
            description=description,
            configuration=configuration,
            created_by='System'
        )
        self.session.add(scenario)
        self.session.commit()
        self.session.refresh(scenario)
        
        print(f"✓ Created scenario: {name}")
        return scenario
    
    def run_scenario(self, scenario_id: int) -> Dict:
        """
        Execute a what-if scenario
        
        Example scenarios:
        - What if we add 5 more judges?
        - What if courtroom capacity increases by 20%?
        - What if we prioritize criminal cases?
        """
        scenario = self.session.query(Scenario).filter(Scenario.id == scenario_id).first()
        if not scenario:
            raise ValueError(f"Scenario {scenario_id} not found")
        
        print(f"\n{'=' * 80}")
        print(f"RUNNING WHAT-IF SCENARIO: {scenario.name}")
        print(f"{'=' * 80}")
        
        config = scenario.configuration
        
        # Apply scenario modifications
        modified_state = self._apply_scenario_config(config)
        
        # Run simulation/optimization with modified state
        results = self._simulate_scheduling(modified_state)
        
        # Store results
        scenario_result = ScenarioResult(
            scenario_id=scenario_id,
            metrics=results,
            analysis=self._generate_analysis(results)
        )
        self.session.add(scenario_result)
        self.session.commit()
        
        print(f"\n✓ Scenario analysis complete")
        return results
    
    def _apply_scenario_config(self, config: Dict) -> Dict:
        """Apply scenario configuration to get modified state"""
        modified_state = {
            'judges': [],
            'courtrooms': [],
            'cases': []
        }
        
        # Get base data
        judges = self.db_ops.get_all_judges()
        courtrooms = self.db_ops.get_all_courtrooms()
        cases = self.db_ops.get_all_cases()
        
        # Apply modifications based on configuration
        if 'add_judges' in config:
            # Simulate adding judges
            num_to_add = config['add_judges']
            modified_state['judges'] = judges + self._generate_dummy_judges(num_to_add)
        else:
            modified_state['judges'] = judges
        
        if 'courtroom_capacity_multiplier' in config:
            # Modify courtroom capacities
            multiplier = config['courtroom_capacity_multiplier']
            for room in courtrooms:
                room.capacity = int(room.capacity * multiplier)
            modified_state['courtrooms'] = courtrooms
        else:
            modified_state['courtrooms'] = courtrooms
        
        if 'priority_filter' in config:
            # Filter cases by priority
            priority = config['priority_filter']
            modified_state['cases'] = [c for c in cases if c.priority == priority]
        else:
            modified_state['cases'] = cases
        
        return modified_state
    
    def _generate_dummy_judges(self, count: int) -> List:
        """Generate dummy judges for scenario"""
        from database_models import Judge
        dummy_judges = []
        for i in range(count):
            judge = Judge(
                judge_id=f"SCENARIO_JDG_{i+1:03d}",
                name=f"Scenario Judge {i+1}",
                experience_years=10,
                max_cases_per_day=10,
                is_active=True
            )
            dummy_judges.append(judge)
        return dummy_judges
    
    def _simulate_scheduling(self, modified_state: Dict) -> Dict:
        """Simulate scheduling with modified state"""
        # Simplified simulation - in real implementation, run full GA
        num_cases = len(modified_state['cases'])
        num_judges = len(modified_state['judges'])
        num_courtrooms = len(modified_state['courtrooms'])
        
        # Calculate theoretical metrics
        theoretical_capacity = num_judges * 10  # 10 cases per judge per day
        utilization = min(100, (num_cases / theoretical_capacity) * 100) if theoretical_capacity > 0 else 0
        
        avg_waiting_time = max(0, 45 - (num_judges * 2))  # Rough estimate
        
        return {
            'total_cases': num_cases,
            'total_judges': num_judges,
            'total_courtrooms': num_courtrooms,
            'theoretical_capacity': theoretical_capacity,
            'utilization_percentage': utilization,
            'estimated_avg_waiting_days': avg_waiting_time,
            'feasible': num_cases <= theoretical_capacity
        }
    
    def _generate_analysis(self, results: Dict) -> str:
        """Generate human-readable analysis"""
        analysis = []
        
        if results['feasible']:
            analysis.append("✓ Scenario is FEASIBLE with given resources.")
        else:
            shortfall = results['total_cases'] - results['theoretical_capacity']
            analysis.append(f"✗ Scenario has CAPACITY SHORTFALL of {shortfall} cases.")
        
        if results['utilization_percentage'] > 90:
            analysis.append("⚠ Resource utilization is very high (>90%).")
        elif results['utilization_percentage'] < 50:
            analysis.append("✓ Resource utilization is healthy (<50%).")
        
        if results['estimated_avg_waiting_days'] > 60:
            analysis.append("⚠ Average waiting time exceeds 60 days.")
        else:
            analysis.append("✓ Average waiting time is acceptable.")
        
        return " ".join(analysis)
    
    def compare_scenarios(self, scenario_ids: List[int]) -> Dict:
        """Compare multiple scenarios"""
        comparison = {
            'scenarios': [],
            'best_scenario': None,
            'comparison_metrics': {}
        }
        
        for scenario_id in scenario_ids:
            scenario = self.session.query(Scenario).filter(Scenario.id == scenario_id).first()
            if scenario:
                results = scenario.results[0] if scenario.results else None
                if results:
                    comparison['scenarios'].append({
                        'id': scenario_id,
                        'name': scenario.name,
                        'metrics': results.metrics
                    })
        
        # Determine best scenario (lowest waiting time, highest utilization)
        if comparison['scenarios']:
            best = min(comparison['scenarios'], 
                      key=lambda x: x['metrics'].get('estimated_avg_waiting_days', 999))
            comparison['best_scenario'] = best
        
        return comparison

# =====================================================================
# HISTORICAL DATA ANALYSIS
# =====================================================================

class HistoricalAnalyzer:
    """Analyze historical scheduling data"""
    
    def __init__(self, session: Session):
        self.session = session
        self.db_ops = DatabaseOperations(session)
    
    def analyze_ga_performance(self, days: int = 30) -> Dict:
        """Analyze GA performance over time"""
        from database_models import GenerationRun
        
        cutoff_date = datetime.utcnow() - timedelta(days=days)
        
        runs = self.session.query(GenerationRun).filter(
            GenerationRun.run_timestamp >= cutoff_date,
            GenerationRun.status == 'Completed'
        ).all()
        
        if not runs:
            return {'message': 'No completed runs in specified period'}
        
        # Calculate statistics
        fitness_scores = [run.best_fitness for run in runs if run.best_fitness]
        execution_times = [run.execution_time for run in runs if run.execution_time]
        convergence_gens = [run.convergence_generation for run in runs if run.convergence_generation]
        
        analysis = {
            'total_runs': len(runs),
            'date_range_days': days,
            'avg_fitness': np.mean(fitness_scores) if fitness_scores else 0,
            'best_fitness': max(fitness_scores) if fitness_scores else 0,
            'worst_fitness': min(fitness_scores) if fitness_scores else 0,
            'avg_execution_time': np.mean(execution_times) if execution_times else 0,
            'avg_convergence_generation': np.mean(convergence_gens) if convergence_gens else 0,
            'trend': 'improving' if len(fitness_scores) > 1 and fitness_scores[-1] > fitness_scores[0] else 'stable'
        }
        
        print(f"\n{'=' * 80}")
        print(f"HISTORICAL PERFORMANCE ANALYSIS ({days} days)")
        print(f"{'=' * 80}")
        print(f"Total Runs: {analysis['total_runs']}")
        print(f"Average Fitness: {analysis['avg_fitness']:.6f}")
        print(f"Best Fitness: {analysis['best_fitness']:.6f}")
        print(f"Avg Execution Time: {analysis['avg_execution_time']:.2f}s")
        print(f"Avg Convergence: {analysis['avg_convergence_generation']:.0f} generations")
        print(f"{'=' * 80}\n")
        
        return analysis
    
    def analyze_case_trends(self, months: int = 6) -> Dict:
        """Analyze case filing and resolution trends"""
        from database_models import Case
        
        cutoff_date = datetime.utcnow() - timedelta(days=months * 30)
        
        cases = self.session.query(Case).filter(
            Case.filing_date >= cutoff_date
        ).all()
        
        # Group by month
        monthly_stats = defaultdict(lambda: {'filed': 0, 'completed': 0, 'pending': 0})
        
        for case in cases:
            month_key = case.filing_date.strftime('%Y-%m')
            monthly_stats[month_key]['filed'] += 1
            
            if case.status == 'Completed':
                monthly_stats[month_key]['completed'] += 1
            elif case.status == 'Pending':
                monthly_stats[month_key]['pending'] += 1
        
        return dict(monthly_stats)
    
    def generate_performance_report(self, output_file: str = 'performance_report.json'):
        """Generate comprehensive performance report"""
        report = {
            'generated_at': datetime.utcnow().isoformat(),
            'ga_performance': self.analyze_ga_performance(30),
            'case_trends': self.analyze_case_trends(6),
            'resource_utilization': self._analyze_resource_utilization()
        }
        
        with open(output_file, 'w') as f:
            json.dump(report, f, indent=2)
        
        print(f"✓ Performance report saved to {output_file}")
        return report
    
    def _analyze_resource_utilization(self) -> Dict:
        """Analyze judge and courtroom utilization"""
        from database_models import Schedule
        
        # Get schedules from last 30 days
        start_date = datetime.utcnow() - timedelta(days=30)
        schedules = self.db_ops.get_schedules_by_date(start_date, datetime.utcnow())
        
        if not schedules:
            return {'message': 'No schedules in specified period'}
        
        # Calculate utilization
        unique_judges = set(s.judge_id for s in schedules)
        unique_courtrooms = set(s.courtroom_id for s in schedules)
        
        total_judges = self.session.query(Judge).filter(Judge.is_active == True).count()
        total_courtrooms = self.session.query(Courtroom).filter(Courtroom.is_active == True).count()
        
        from database_models import Judge, Courtroom
        
        return {
            'judges_utilized': len(unique_judges),
            'total_judges': total_judges,
            'judge_utilization': (len(unique_judges) / total_judges * 100) if total_judges > 0 else 0,
            'courtrooms_utilized': len(unique_courtrooms),
            'total_courtrooms': total_courtrooms,
            'courtroom_utilization': (len(unique_courtrooms) / total_courtrooms * 100) if total_courtrooms > 0 else 0
        }

# =====================================================================
# EXAMPLE USAGE
# =====================================================================

if __name__ == "__main__":
    DATABASE_URL = "postgresql://postgres:password@localhost:5432/court_scheduler"
    
    db_manager = DatabaseManager(DATABASE_URL)
    session = db_manager.get_session()
    
    print("\n" + "=" * 80)
    print("ADVANCED FEATURES DEMONSTRATION")
    print("=" * 80)
    
    # 1. Real-Time Constraint Updates
    print("\n1. REAL-TIME CONSTRAINT MANAGEMENT")
    constraint_mgr = RealTimeConstraintManager(session)
    
    # Update judge availability
    constraint_mgr.update_judge_availability(
        judge_id="JDG_001",
        date=datetime.now() + timedelta(days=5),
        start_time="09:00",
        end_time="17:00",
        is_available=False,
        reason="Medical leave"
    )
    
    # 2. What-If Scenario Analysis
    print("\n2. WHAT-IF SCENARIO ANALYSIS")
    whatif_analyzer = WhatIfAnalyzer(session)
    
    # Create scenario: Add 5 more judges
    scenario1 = whatif_analyzer.create_scenario(
        name="Add 5 Judges",
        description="Evaluate impact of hiring 5 additional judges",
        configuration={'add_judges': 5}
    )
    
    results1 = whatif_analyzer.run_scenario(scenario1.id)
    print(f"Results: {json.dumps(results1, indent=2)}")
    
    # Create scenario: Increase courtroom capacity
    scenario2 = whatif_analyzer.create_scenario(
        name="Expand Courtrooms",
        description="Increase all courtroom capacities by 30%",
        configuration={'courtroom_capacity_multiplier': 1.3}
    )
    
    # 3. Historical Analysis
    print("\n3. HISTORICAL PERFORMANCE ANALYSIS")
    historical_analyzer = HistoricalAnalyzer(session)
    performance = historical_analyzer.analyze_ga_performance(30)
    
    # Generate comprehensive report
    report = historical_analyzer.generate_performance_report()
    
    # 4. Multi-Objective Optimization (NSGA-II)
    print("\n4. MULTI-OBJECTIVE OPTIMIZATION (NSGA-II)")
    print("Note: Full NSGA-II implementation requires integration with GA scheduler")
    
    nsga2 = NSGAII(population_size=50, max_generations=100)
    # pareto_fronts = nsga2.run()  # Would run full optimization
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)
    
    session.close()
    db_manager.close()