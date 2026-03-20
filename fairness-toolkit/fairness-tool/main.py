from tkinter import messagebox
from .deps import SKLEARN_OK
from .gui import FairnessToolGUI

from .runner import PipelineConfig, run_pipeline

'''
if __name__ == "__main__":
    cfg = PipelineConfig(
        df_or_path="path/to/data.csv",
        target="y",
        protected=["sex", "race"],
        features=["age", "bmi", "hdl", "ldl", "sex", "race"],  # target/protected will be removed automatically
        model_name="logreg",
        model_params={"max_iter": 1000},
        techniques=[
            "Pre:Reweight (y,a)",
            "In:Reductions (EO)",
            "Post:Youden per group",
        ],
        run_baseline=True,
        run_combined=True,
    )

    results = run_pipeline(cfg)
    for rr in results:
        print(rr.name)
        print(rr.overall)
        print(rr.gap_metrics)
        print()
'''


### Below runs the GUI application ###
      
def main():
    if not SKLEARN_OK:
        messagebox.showerror("Missing dependency", "scikit-learn is required. Please install it first.")
        return
    app = FairnessToolGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
