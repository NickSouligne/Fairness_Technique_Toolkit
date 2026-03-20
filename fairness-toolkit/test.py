from .runner import PipelineConfig, run_pipeline

cfg = PipelineConfig(
    df_or_path="C:\\Users\\nicks\\Documents\\Research\\LLM and ML\\synthetic_ehr.csv",          
    target="cardio:mi",
    protected=["sex", "race"],
    features=["age", "bmi", "alcohol_use", "temp"], 
    model_name="XGBoost",
    model_params={"max_iter": 1000},
    techniques=["Post:Reject-Option Kamiran"],
    run_baseline=True,
    run_combined=False,
)

results = run_pipeline(cfg)

for rr in results:
    print("=" * 80)
    print(rr.name)
    print(rr.overall)
