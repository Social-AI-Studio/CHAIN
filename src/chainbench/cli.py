"""
Command-line interface for the CHAIN benchmark (chainbench).

This module provides a comprehensive CLI for running the CHAIN interactive 3D
physics-driven benchmark with VLM agents. It supports automatic component
discovery, configuration management, and flexible execution modes.
"""

import argparse
import sys
import os
import json
from typing import Optional, Dict, List, Any
from pathlib import Path

from chainbench.core.config import Config, load_config, create_default_config, validate_config
from chainbench.core.base import TaskDifficulty
from chainbench.core.registry import ENVIRONMENT_REGISTRY, TASK_REGISTRY, AGENT_REGISTRY
from chainbench.runner import BenchmarkRunner
from chainbench.evaluation.evaluator import BenchmarkEvaluator
from chainbench.evaluation.metrics import MetricsCalculator
from chainbench.utils.display import StatusDisplay, LiveLogger


def get_available_components() -> Dict[str, List[str]]:
    """Get dynamically registered components."""
    # Import all modules to trigger registration
    try:
        import chainbench.tasks
        import chainbench.environment
        import chainbench.agents
    except ImportError:
        pass
    
    return {
        "tasks": list(TASK_REGISTRY.keys()),
        "environments": list(ENVIRONMENT_REGISTRY.keys()),
        "agents": list(AGENT_REGISTRY.keys())
    }


def create_parser() -> argparse.ArgumentParser:
    """Create command-line argument parser."""
    components = get_available_components()
    difficulties = [d.value for d in TaskDifficulty]
    
    parser = argparse.ArgumentParser(
        description="chainbench: CHAIN – Causal Hierarchy of Actions and Interactions Benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=f"""
Examples:
  # Run single task
  chainbench run --config eval_configs/luban.yaml
  
  # Run benchmark with multiple trials  
  chainbench benchmark --config eval_configs/luban.yaml --num-runs 5
  
  # Evaluate existing results
  chainbench evaluate --results-dir logs/
  
  # Create default configuration
  chainbench create-config --output config.yaml --task-type luban_disassembly
  
  # List available components
  chainbench list-components
  
  # Show component details
  chainbench show-component --type task --name luban_disassembly

Available Components:
  Tasks: {', '.join(components['tasks']) if components['tasks'] else 'None registered'}
  Environments: {', '.join(components['environments']) if components['environments'] else 'None registered'}
  Agents: {', '.join(components['agents']) if components['agents'] else 'None registered'}
  Difficulties: {', '.join(difficulties)}
        """
    )
    
    subparsers = parser.add_subparsers(dest="command", help="Available commands")
    
    # Run command
    run_parser = subparsers.add_parser("run", help="Run single task instance")
    run_parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    run_parser.add_argument("--gui", action="store_true", help="Enable physics simulation GUI")
    run_parser.add_argument("--task-type", choices=components['tasks'], help="Override task type")
    run_parser.add_argument("--difficulty", choices=difficulties, help="Override difficulty")
    run_parser.add_argument("--model", help="Override model name")
    run_parser.add_argument("--output-dir", help="Override output directory")
    run_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Benchmark command
    benchmark_parser = subparsers.add_parser("benchmark", help="Run complete benchmark")
    benchmark_parser.add_argument("--config", "-c", required=True, help="Path to configuration file")
    benchmark_parser.add_argument("--num-runs", "-n", type=int, default=1, help="Number of task runs")
    benchmark_parser.add_argument("--gui", action="store_true", help="Enable physics simulation GUI")
    benchmark_parser.add_argument("--model", help="Override model name")
    benchmark_parser.add_argument("--output-dir", help="Override output directory")
    benchmark_parser.add_argument("--verbose", "-v", action="store_true", help="Enable verbose output")
    
    # Evaluate command
    eval_parser = subparsers.add_parser("evaluate", help="Evaluate existing results")
    eval_parser.add_argument("--results-dir", required=True, help="Directory containing result files")
    eval_parser.add_argument("--output", help="Output file for evaluation results")
    eval_parser.add_argument("--compare-models", nargs="+", help="Compare specific models")
    eval_parser.add_argument("--format", choices=["excel", "json", "csv"], default="excel", help="Output format")
    
    # Create config command
    config_parser = subparsers.add_parser("create-config", help="Create default configuration file")
    config_parser.add_argument("--output", "-o", default="config.yaml", help="Output configuration file")
    config_parser.add_argument("--task-type", choices=components['tasks'] if components['tasks'] else ["luban_disassembly"], 
                              default=components['tasks'][0] if components['tasks'] else "luban_disassembly", help="Default task type")
    config_parser.add_argument("--environment-type", choices=components['environments'] if components['environments'] else ["luban"], 
                              default=components['environments'][0] if components['environments'] else "luban", help="Default environment type")
    config_parser.add_argument("--agent-type", choices=components['agents'] if components['agents'] else ["openai"], 
                              default=components['agents'][0] if components['agents'] else "openai", help="Default agent type")
    config_parser.add_argument("--difficulty", choices=difficulties, default="easy", help="Default difficulty")
    config_parser.add_argument("--model", default="gpt-4o", help="Default model name")
    
    # Validate config command
    validate_parser = subparsers.add_parser("validate-config", help="Validate configuration file")
    validate_parser.add_argument("config", help="Configuration file to validate")
    validate_parser.add_argument("--strict", action="store_true", help="Treat warnings as errors")
    
    # List components command
    list_parser = subparsers.add_parser("list-components", help="List available components")
    list_parser.add_argument("--type", choices=["tasks", "environments", "agents", "all"], default="all", help="Component type to list")
    list_parser.add_argument("--format", choices=["table", "json"], default="table", help="Output format")
    
    # Show component command
    show_parser = subparsers.add_parser("show-component", help="Show detailed information about a component")
    show_parser.add_argument("--type", choices=["task", "environment", "agent"], required=True, help="Component type")
    show_parser.add_argument("--name", required=True, help="Component name")
    
    # Init command for setting up environment
    init_parser = subparsers.add_parser("init", help="Initialize chainbench environment")
    init_parser.add_argument("--project-name", default="my_chainbench_project", help="Project name")
    init_parser.add_argument("--force", action="store_true", help="Force initialization even if files exist")
    
    return parser


def run_command(args) -> int:
    """Execute run command."""
    logger = LiveLogger(verbose=getattr(args, 'verbose', False))
    
    try:
        StatusDisplay.print_header("chainbench Task Execution")
        
        # Load and validate configuration
        config = _load_and_validate_config(args, logger)
        if config is None:
            return 1
            
        # Apply command line overrides
        _apply_run_overrides(config, args)
        
        # Display configuration summary
        _display_run_config(config)
        
        # Create and run benchmark
        logger.log_action("Creating benchmark runner")
        runner = BenchmarkRunner(config)
        runner.setup()
        logger.log_result("Runner initialized")
        
        # Execute task
        result = runner.run_single_task()
        
        # Display and handle results
        return _handle_run_results(result, logger)
        
    except KeyboardInterrupt:
        logger.log_warning("Task execution interrupted by user")
        return 1
    except Exception as e:
        logger.log_error(f"Failed to run task: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            logger.log_error(traceback.format_exc())
        return 1


def _load_and_validate_config(args, logger) -> Optional[Config]:
    """Load and validate configuration with error handling."""
    try:
        logger.log_action("Loading configuration")
        config = load_config(args.config)
        logger.log_result("Configuration loaded")
        
        logger.log_action("Validating configuration")
        issues = validate_config(config)
        
        errors = [issue for issue in issues if issue.startswith("ERROR")]
        warnings = [issue for issue in issues if not issue.startswith("ERROR")]
        
        if errors:
            StatusDisplay.print_section("Configuration Errors")
            for error in errors:
                logger.log_error(error.replace("ERROR: ", ""))
            return None
        elif warnings:
            StatusDisplay.print_section("Configuration Warnings")
            for warning in warnings:
                logger.log_warning(warning)
        
        logger.log_result("Configuration validation completed")
        return config
        
    except FileNotFoundError:
        logger.log_error(f"Configuration file not found: {args.config}")
        logger.log_info("Use 'chainbench create-config' to create a default configuration")
        return None
    except Exception as e:
        logger.log_error(f"Configuration error: {e}")
        return None


def _apply_run_overrides(config: Config, args) -> None:
    """Apply command line overrides to configuration."""
    if hasattr(args, 'gui') and args.gui:
        config.environment.gui = True
    if hasattr(args, 'task_type') and args.task_type:
        config.task.type = args.task_type
    if hasattr(args, 'difficulty') and args.difficulty:
        config.task.difficulty = TaskDifficulty(args.difficulty)
    if hasattr(args, 'model') and args.model:
        config.agent.model_name = args.model
    if hasattr(args, 'output_dir') and args.output_dir:
        config.runner.log_dir = args.output_dir


def _display_run_config(config: Config) -> None:
    """Display configuration summary."""
    config_info = {
        "Task Type": config.task.type,
        "Task Name": config.task.name,
        "Difficulty": config.task.difficulty.value if config.task.difficulty else "unknown",
        "Environment": config.environment.type,
        "Agent": f"{config.agent.type} ({config.agent.model_name})",
        "Output Directory": config.runner.log_dir,
        "GUI Mode": "Enabled" if config.environment.gui else "Disabled"
    }
    StatusDisplay.print_config(config_info, "Execution Configuration")


def _handle_run_results(result, logger) -> int:
    """Handle and display task execution results."""
    task_results = {
        "Task ID": result.task_id,
        "Task Type": result.task_type,
        "Execution Time": f"{result.execution_time:.2f}s",
        "Steps Taken": result.total_steps,
        "Success": "✓" if result.success else "✗"
    }
    
    if hasattr(result, 'metadata') and result.metadata:
        if result.metadata.get('total_tokens'):
            task_results["Total Tokens"] = result.metadata['total_tokens']
        if result.metadata.get('difficulty'):
            task_results["Difficulty"] = result.metadata['difficulty']
            
    StatusDisplay.print_results(task_results, "Task Execution Results")
    
    if result.error_message:
        logger.log_error(f"Task failed: {result.error_message}")
        return 1
    elif result.success:
        logger.log_result("Task completed successfully!")
        return 0
    else:
        logger.log_warning("Task completed but success criteria not met")
        return 0


def benchmark_command(args) -> int:
    """Execute benchmark command."""
    logger = LiveLogger(verbose=getattr(args, 'verbose', False))
    
    try:
        StatusDisplay.print_header(f"chainbench Benchmark - {args.num_runs} Run{'s' if args.num_runs > 1 else ''}")
        
        # Load and validate configuration
        config = _load_and_validate_config(args, logger)
        if config is None:
            return 1
            
        # Apply command line overrides
        _apply_benchmark_overrides(config, args)
        
        # Display configuration summary
        _display_benchmark_config(config, args.num_runs)
        
        # Create and run benchmark
        logger.log_action("Creating benchmark runner")
        runner = BenchmarkRunner(config)
        runner.setup()
        logger.log_result("Runner initialized")
        
        # Execute benchmark
        logger.log_info(f"Starting benchmark with {args.num_runs} run{'s' if args.num_runs > 1 else ''}")
        evaluation_result = runner.run_benchmark(num_runs=args.num_runs)
        
        # Display final results
        _display_benchmark_summary(evaluation_result, config, logger)
        
        return 0
        
    except KeyboardInterrupt:
        logger.log_warning("Benchmark execution interrupted by user")
        return 1
    except Exception as e:
        logger.log_error(f"Failed to run benchmark: {e}")
        if getattr(args, 'verbose', False):
            import traceback
            logger.log_error(traceback.format_exc())
        return 1


def _apply_benchmark_overrides(config: Config, args) -> None:
    """Apply command line overrides for benchmark."""
    if hasattr(args, 'gui') and args.gui:
        config.environment.gui = True
    if hasattr(args, 'model') and args.model:
        config.agent.model_name = args.model
    if hasattr(args, 'output_dir') and args.output_dir:
        config.runner.log_dir = args.output_dir


def _display_benchmark_config(config: Config, num_runs: int) -> None:
    """Display benchmark configuration summary."""
    config_info = {
        "Task Type": config.task.type,
        "Task Name": config.task.name,
        "Difficulty": config.task.difficulty.value if config.task.difficulty else "unknown",
        "Environment": config.environment.type,
        "Agent": f"{config.agent.type} ({config.agent.model_name})",
        "Number of Runs": num_runs,
        "Output Directory": config.runner.log_dir,
        "GUI Mode": "Enabled" if config.environment.gui else "Disabled"
    }
    StatusDisplay.print_config(config_info, "Benchmark Configuration")


def _display_benchmark_summary(evaluation_result, config: Config, logger: LiveLogger) -> None:
    """Display final benchmark summary."""
    if evaluation_result:
        summary_info = {
            "Model": config.agent.model_name,
            "Task": f"{config.task.name} ({config.task.difficulty.value if config.task.difficulty else 'unknown'})",
            "Total Tasks": len(evaluation_result.task_results),
            "Successful": sum(1 for r in evaluation_result.task_results if r.success),
            "Accuracy": f"{evaluation_result.accuracy:.1%}"
        }
        
        if evaluation_result.pass_at_k:
            for k, rate in evaluation_result.pass_at_k.items():
                summary_info[f"Pass@{k}"] = f"{rate:.1%}"
        
        StatusDisplay.print_results(summary_info, "Benchmark Summary")
        
        # Log output files
        results_path = Path(config.runner.log_dir) / config.runner.experiment_name / config.runner.results_excel_path
        logger.log_info(f"Results saved to: {results_path}")
    else:
        logger.log_warning("No evaluation results available")


def evaluate_command(args) -> int:
    """Execute evaluate command."""
    logger = LiveLogger(verbose=True)
    
    try:
        StatusDisplay.print_header("Results Evaluation")
        
        results_dir = Path(args.results_dir)
        if not results_dir.exists():
            logger.log_error(f"Results directory not found: {results_dir}")
            logger.log_info("Make sure you have run some benchmarks first")
            return 1
        
        logger.log_action("Scanning results directory")
        
        # Find various types of result files
        result_patterns = [
            "**/experiment_results.xlsx",
            "**/benchmark_result.json", 
            "**/*_evaluation_report.json",
            "**/task_results.json"
        ]
        
        all_result_files = []
        for pattern in result_patterns:
            all_result_files.extend(results_dir.glob(pattern))
        
        if not all_result_files:
            logger.log_error(f"No result files found in {results_dir}")
            logger.log_info("Expected file patterns: " + ", ".join(result_patterns))
            return 1
            
        logger.log_result(f"Found {len(all_result_files)} result files")
        
        # Display summary and process results
        files_info = {
            "Excel Files": len([f for f in all_result_files if f.suffix == '.xlsx']),
            "JSON Files": len([f for f in all_result_files if f.suffix == '.json']),
            "Total Files": len(all_result_files)
        }
        StatusDisplay.print_results(files_info, "Found Results")
        
        logger.log_info("Results evaluation completed successfully")
        return 0
        
    except Exception as e:
        logger.log_error(f"Failed to evaluate results: {e}")
        return 1


def create_config_command(args) -> int:
    """Execute create-config command."""
    logger = LiveLogger(verbose=True)
    
    try:
        StatusDisplay.print_header("Creating Configuration File")
        
        # Check if file exists and handle overwrite
        if Path(args.output).exists():
            logger.log_warning(f"Configuration file already exists: {args.output}")
            response = input("Overwrite existing file? (y/N): ").strip().lower()
            if response not in ['y', 'yes']:
                logger.log_info("Configuration creation cancelled")
                return 0
        
        logger.log_action("Creating configuration")
        config = _create_custom_config(args)
        
        # Save configuration
        import yaml
        with open(args.output, 'w', encoding='utf-8') as f:
            yaml.dump(config.to_dict(), f, default_flow_style=False, indent=2)
        
        logger.log_result(f"Configuration created: {args.output}")
        
        # Display configuration summary
        config_info = {
            "Output File": args.output,
            "Task Type": args.task_type,
            "Environment Type": args.environment_type,
            "Agent Type": args.agent_type,
            "Model": args.model,
            "Difficulty": args.difficulty
        }
        StatusDisplay.print_results(config_info, "Configuration Summary")
        
        # Provide next steps
        logger.log_info("\nNext steps:")
        logger.log_info("1. Edit the configuration file to set your API keys")
        logger.log_info("2. Validate the configuration: chainbench validate-config " + args.output)
        logger.log_info("3. Run a test: chainbench run --config " + args.output)
        
        return 0
        
    except Exception as e:
        logger.log_error(f"Failed to create config: {e}")
        return 1


def _create_custom_config(args) -> Config:
    """Create customized configuration based on arguments."""
    from chainbench.core.config import (
        Config, RunnerConfig, AgentConfig, JudgementConfig, 
        EnvironmentConfig, TaskConfig
    )
    
    # Create runner config
    runner = RunnerConfig(
        experiment_name=f"{args.task_type}_{args.difficulty}_experiment",
        log_dir="logs",
        results_excel_path="experiment_results.xlsx"
    )
    
    # Create agent config
    agent = AgentConfig(
        type=args.agent_type,
        model_name=args.model,
        api_key=None,  # Will be loaded from environment
        base_url=None,  # Will be loaded from environment
        temperature=0.7,
        max_tokens=500
    )
    
    # Create judgement config
    judgement = JudgementConfig(
        type=args.agent_type,
        model_name=args.model,
        temperature=0.1,
        max_tokens=50
    )
    
    # Create environment config
    environment = EnvironmentConfig(
        type=args.environment_type,
        gui=False,
        render_width=512,
        render_height=512,
        max_steps=50
    )
    
    # Create task config
    task = TaskConfig(
        type=args.task_type,
        name=f"{args.task_type}_{args.difficulty}",
        difficulty=TaskDifficulty(args.difficulty)
    )
    
    return Config(
        runner=runner,
        agent=agent,
        judgement=judgement,
        environment=environment,
        task=task
    )


def validate_config_command(args) -> int:
    """Execute validate-config command."""
    logger = LiveLogger(verbose=True)
    
    try:
        StatusDisplay.print_header("Configuration Validation")
        
        config_path = Path(args.config)
        if not config_path.exists():
            logger.log_error(f"Configuration file not found: {args.config}")
            return 1
        
        logger.log_action(f"Loading configuration from: {args.config}")
        config = load_config(args.config)
        logger.log_result("Configuration loaded successfully")
        
        # Display configuration summary
        _display_config_summary(config)
        
        logger.log_action("Validating configuration")
        issues = validate_config(config)
        
        errors = [issue for issue in issues if issue.startswith("ERROR")]
        warnings = [issue for issue in issues if not issue.startswith("ERROR")]
        
        # Handle strict mode
        if hasattr(args, 'strict') and args.strict and warnings:
            errors.extend(warnings)
            warnings = []
        
        if errors:
            StatusDisplay.print_section("❌ Configuration Errors")
            for i, error in enumerate(errors, 1):
                logger.log_error(f"{i}. {error.replace('ERROR: ', '')}")
            
            summary = {
                "Status": "❌ FAILED",
                "Errors Found": len(errors),
                "Warnings Found": len(warnings)
            }
            StatusDisplay.print_results(summary, "Validation Summary")
            logger.log_error("Configuration validation failed. Please fix the errors above.")
            return 1
            
        elif warnings:
            StatusDisplay.print_section("⚠️  Configuration Warnings")
            for i, warning in enumerate(warnings, 1):
                logger.log_warning(f"{i}. {warning}")
            
            summary = {
                "Status": "✓ VALID (with warnings)",
                "Warnings Found": len(warnings)
            }
            StatusDisplay.print_results(summary, "Validation Summary")
            logger.log_info("Configuration is valid but has warnings. Consider addressing them.")
            return 0
        else:
            summary = {
                "Status": "✅ PERFECT",
                "Errors Found": 0,
                "Warnings Found": 0
            }
            StatusDisplay.print_results(summary, "Validation Summary")
            logger.log_result("✨ Configuration is perfect! Ready to run.")
            return 0
            
    except FileNotFoundError:
        logger.log_error(f"Configuration file not found: {args.config}")
        return 1
    except Exception as e:
        logger.log_error(f"Failed to validate config: {e}")
        return 1


def _display_config_summary(config: Config) -> None:
    """Display configuration summary."""
    config_info = {
        "Experiment": config.runner.experiment_name,
        "Task": f"{config.task.type} ({config.task.name})",
        "Difficulty": config.task.difficulty.value if config.task.difficulty else "unknown",
        "Environment": config.environment.type,
        "Agent": f"{config.agent.type} ({config.agent.model_name})",
        "Output Dir": config.runner.log_dir
    }
    StatusDisplay.print_config(config_info, "Configuration Overview")


def list_components_command(args) -> int:
    """Execute list-components command."""
    logger = LiveLogger(verbose=False)
    
    try:
        components = get_available_components()
        component_type = args.type
        output_format = getattr(args, 'format', 'table')
        
        if component_type == "all":
            if output_format == "json":
                print(json.dumps(components, indent=2))
            else:
                StatusDisplay.print_header("Available Components")
                for comp_type, comp_list in components.items():
                    StatusDisplay.print_section(f"{comp_type.title()}")
                    for comp in comp_list:
                        print(f"  • {comp}")
        else:
            comp_list = components.get(component_type, [])
            if output_format == "json":
                print(json.dumps({component_type: comp_list}, indent=2))
            else:
                StatusDisplay.print_header(f"Available {component_type.title()}")
                for comp in comp_list:
                    print(f"  • {comp}")
        
        return 0
        
    except Exception as e:
        logger.log_error(f"Failed to list components: {e}")
        return 1


def show_component_command(args) -> int:
    """Execute show-component command."""
    logger = LiveLogger(verbose=False)
    
    try:
        comp_type = args.type
        comp_name = args.name
        
        # Get registry based on type
        registry_map = {
            "task": TASK_REGISTRY,
            "environment": ENVIRONMENT_REGISTRY,
            "agent": AGENT_REGISTRY
        }
        
        registry = registry_map.get(comp_type)
        if not registry:
            logger.log_error(f"Unknown component type: {comp_type}")
            return 1
        
        component_class = registry.get(comp_name)
        if not component_class:
            logger.log_error(f"{comp_type.title()} '{comp_name}' not found")
            available = list(registry.keys())
            logger.log_info(f"Available {comp_type}s: {', '.join(available)}")
            return 1
        
        # Display component information
        StatusDisplay.print_header(f"{comp_type.title()}: {comp_name}")
        
        info = {
            "Name": comp_name,
            "Type": comp_type,
            "Class": component_class.__name__,
            "Module": component_class.__module__
        }
        
        if hasattr(component_class, '__doc__') and component_class.__doc__:
            info["Description"] = component_class.__doc__.strip().split('\n')[0]
        
        StatusDisplay.print_results(info, "Component Details")
        
        # Show docstring if available
        if hasattr(component_class, '__doc__') and component_class.__doc__:
            StatusDisplay.print_section("Documentation")
            print(component_class.__doc__.strip())
        
        return 0
        
    except Exception as e:
        logger.log_error(f"Failed to show component: {e}")
        return 1


def init_command(args) -> int:
    """Execute init command to set up project environment."""
    logger = LiveLogger(verbose=True)
    
    try:
        StatusDisplay.print_header("Initializing chainbench Project")
        
        project_name = args.project_name
        project_dir = Path(project_name)
        
        # Check if directory exists
        if project_dir.exists() and not args.force:
            logger.log_error(f"Directory '{project_name}' already exists")
            logger.log_info("Use --force to overwrite existing directory")
            return 1
        
        # Create project structure
        logger.log_action(f"Creating project directory: {project_name}")
        project_dir.mkdir(exist_ok=True)
        
        (project_dir / "configs").mkdir(exist_ok=True)
        (project_dir / "logs").mkdir(exist_ok=True)
        (project_dir / "results").mkdir(exist_ok=True)
        
        # Create sample configurations
        _create_sample_configs(project_dir / "configs", logger)
        
        # Create .env template
        _create_env_template(project_dir)
        
        # Create README
        _create_project_readme(project_dir, project_name)
        
        logger.log_result("Project structure created")
        
        # Display summary and next steps
        summary = {
            "Project Name": project_name,
            "Config Files": "3 sample configurations created",
            "Environment File": ".env template created",
            "Documentation": "README.md created"
        }
        StatusDisplay.print_results(summary, "Project Initialization Complete")
        
        logger.log_info(f"\n📁 Project created: {project_dir.absolute()}")
        logger.log_info("\n🚀 Next steps:")
        logger.log_info(f"   cd {project_name}")
        logger.log_info("   cp .env .env.local")
        logger.log_info("   # Edit .env.local with your API keys")
        logger.log_info("   chainbench validate-config configs/luban_easy.yaml")
        logger.log_info("   chainbench run --config configs/luban_easy.yaml")
        
        return 0
        
    except Exception as e:
        logger.log_error(f"Failed to initialize project: {e}")
        return 1


def _create_sample_configs(configs_dir: Path, logger: LiveLogger) -> None:
    """Create sample configuration files."""
    import yaml
    
    components = get_available_components()
    
    # Create a basic luban config
    luban_config = {
        "runner": {
            "experiment_name": "luban_easy_test",
            "log_dir": "logs",
            "results_excel_path": "results/luban_results.xlsx"
        },
        "agent": {
            "type": components['agents'][0] if components['agents'] else "openai",
            "model_name": "gpt-4o",
            "temperature": 0.6,
            "max_tokens": 4096
        },
        "judgement": {
            "type": components['agents'][0] if components['agents'] else "openai",
            "model_name": "gpt-4o-mini",
            "temperature": 0.1,
            "max_tokens": 2048
        },
        "environment": {
            "type": components['environments'][0] if components['environments'] else "luban",
            "gui": False,
            "max_steps": 6
        },
        "task": {
            "type": components['tasks'][0] if components['tasks'] else "luban_disassembly",
            "name": "luban_easy_demo",
            "difficulty": "easy"
        }
    }
    
    with open(configs_dir / "luban_easy.yaml", 'w') as f:
        yaml.dump(luban_config, f, default_flow_style=False, indent=2)
    
    logger.log_result("Sample configurations created")


def _create_env_template(project_dir: Path) -> None:
    """Create .env template file."""
    env_content = """# API Keys for chainbench
# Copy this file and fill in your actual API keys

# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_BASE_URL=https://api.openai.com/v1

# Optional: Custom model endpoints
# CUSTOM_MODEL_API_KEY=your_custom_api_key
# CUSTOM_MODEL_BASE_URL=https://your-custom-endpoint.com/v1
"""
    
    env_file = project_dir / ".env"
    with open(env_file, 'w') as f:
        f.write(env_content)


def _create_project_readme(project_dir: Path, project_name: str) -> None:
    """Create project README.md file."""
    readme_content = f"""# {project_name}

CHAIN benchmark project for interactive 3D physics-driven VLM evaluation.

## Setup

1. Copy `.env` file and fill in your API keys:
   ```bash
   cp .env .env.local
   # Edit .env.local with your actual API keys
   ```

2. Validate a configuration:
   ```bash
   chainbench validate-config configs/luban_easy.yaml
   ```

3. Run a test:
   ```bash
   chainbench run --config configs/luban_easy.yaml
   ```

## Available Commands

- `chainbench list-components` - See available tasks, environments, agents
- `chainbench show-component --type task --name luban_disassembly` - Get details about components
- `chainbench create-config --output my_config.yaml` - Generate new configurations
- `chainbench benchmark --config configs/luban_easy.yaml --num-runs 5` - Run benchmarks
- `chainbench evaluate --results-dir logs/` - Analyze results

## Results

All results will be saved to the `logs/` directory, with Excel reports in `results/`.
"""
    
    readme_file = project_dir / "README.md"
    with open(readme_file, 'w') as f:
        f.write(readme_content)


def main() -> int:
    """Main CLI entry point."""
    try:
        parser = create_parser()
        
        if len(sys.argv) == 1:
            parser.print_help()
            return 1
            
        args = parser.parse_args()
        
        # Route to appropriate command handler
        command_handlers = {
            "run": run_command,
            "benchmark": benchmark_command,
            "evaluate": evaluate_command,
            "create-config": create_config_command,
            "validate-config": validate_config_command,
            "list-components": list_components_command,
            "show-component": show_component_command,
            "init": init_command
        }
        
        handler = command_handlers.get(args.command)
        if handler:
            return handler(args)
        else:
            parser.print_help()
            return 1
            
    except KeyboardInterrupt:
        print("\n⚠️  Interrupted by user")
        return 1
    except Exception as e:
        print(f"\n❌ Unexpected error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
