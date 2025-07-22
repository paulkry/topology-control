from src.CPipelineOrchestrator import CPipelineOrchestrator

if __name__ == "__main__":
    
    # Run full pipeline
    orchestrator = CPipelineOrchestrator()
    orchestrator.run_full_pipeline()