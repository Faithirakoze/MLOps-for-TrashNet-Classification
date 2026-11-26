from src.mlops_pipeline import run_mlops_pipeline


if __name__ == "__main__":
    results = run_mlops_pipeline()

    print("\nFINAL PIPELINE RESULTS")
    print(results)