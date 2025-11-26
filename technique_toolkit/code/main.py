from tkinter import messagebox
from deps import SKLEARN_OK
from gui import FairnessToolGUI


def main():
    if not SKLEARN_OK:
        messagebox.showerror("Missing dependency", "scikit-learn is required. Please install it first.")
        return
    app = FairnessToolGUI()
    app.mainloop()


if __name__ == "__main__":
    main()
