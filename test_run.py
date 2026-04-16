from pathlib import Path
from dotenv import load_dotenv

def run_pipeline_locally():
    from src.tracing.langfuse_helpers import generate_session_id
    from src.challanges.level_1.runner import run_level_1

    load_dotenv()
    print("--- Starting Local Run ---")
    
    test_dir = Path("datasets/test_fixtures")
    output_file = Path("test_output.txt")
    
    if not test_dir.exists():
        print(f"Error: Could not find {test_dir}. Please create it and add a tiny transactions.csv")
        return

    session_id = generate_session_id()
    print(f"Generated Test Session ID: {session_id}")

    result = run_level_1(
        input_dir=test_dir,
        output=output_file,
        session_id=session_id,
    )
    
    print("\nPipeline completed successfully!")
    print(f"Detected Fraud IDs: {result.suspected_transaction_ids}")
    
    if output_file.exists():
        print(f"Output file successfully generated at: {result.output_path}")
        with open(output_file, 'r') as f:
            print("File contents:")
            print(f.read())
            
        # output_file.unlink()
        print("Cleaned up test_output.txt")

if __name__ == "__main__":
    run_pipeline_locally()
