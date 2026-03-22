import csv
import subprocess
import tkinter as tk
from tkinter import ttk, messagebox
from pathlib import Path

DATA_FILE = Path("stocks.csv")
RESULTS_FILE = Path("results.csv")
BACKEND_EXE = "./solver_app"


def load_stocks():
    stocks = []
    with open(DATA_FILE, newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stocks.append({
                "ticker": row["ticker"],
                "price": float(row["price"]),
                "growth_rate": float(row["growth_rate"]),
                "dividend_yield": float(row["dividend_yield"]),
            })
    return stocks


class PortfolioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CUDA Dividend Portfolio Allocation Optimizer")
        self.root.geometry("900x650")

        self.stocks = load_stocks()
        self.selected = {}
        self._build_ui()

    def _build_ui(self):
        title = ttk.Label(
            self.root,
            text="CUDA Dividend Portfolio Allocation Optimizer",
            font=("Arial", 16, "bold"),
        )
        title.pack(pady=10)

        controls = ttk.Frame(self.root, padding=10)
        controls.pack(fill="x")

        ttk.Label(controls, text="Monthly Investment ($):").grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.monthly_var = tk.StringVar(value="500")
        ttk.Entry(controls, textvariable=self.monthly_var, width=12).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(controls, text="Duration (years):").grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.years_var = tk.StringVar(value="10")
        ttk.Entry(controls, textvariable=self.years_var, width=12).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(controls, text="Goal:").grid(row=0, column=4, sticky="w", padx=5, pady=5)
        self.goal_var = tk.StringVar(value="balanced")
        ttk.Combobox(
            controls,
            textvariable=self.goal_var,
            values=["growth", "income", "balanced"],
            state="readonly",
            width=12,
        ).grid(row=0, column=5, padx=5, pady=5)

        stock_frame = ttk.LabelFrame(self.root, text="Select Stocks", padding=10)
        stock_frame.pack(fill="both", expand=False, padx=10, pady=10)

        for i, stock in enumerate(self.stocks):
            var = tk.BooleanVar(value=(i < 4))
            self.selected[stock["ticker"]] = var
            text = (
                f'{stock["ticker"]} | Price ${stock["price"]:.2f} | '
                f'Growth {stock["growth_rate"]*100:.1f}% | '
                f'Dividend {stock["dividend_yield"]*100:.1f}%'
            )
            ttk.Checkbutton(stock_frame, text=text, variable=var).grid(
                row=i, column=0, sticky="w", padx=5, pady=3
            )

        button_frame = ttk.Frame(self.root, padding=10)
        button_frame.pack(fill="x")

        ttk.Button(button_frame, text="Run Optimization", command=self.run_backend).pack(side="left", padx=5)
        ttk.Button(button_frame, text="Clear Output", command=self.clear_output).pack(side="left", padx=5)

        output_frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        output_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.output = tk.Text(output_frame, wrap="word", font=("Courier New", 10))
        self.output.pack(fill="both", expand=True)

    def clear_output(self):
        self.output.delete("1.0", tk.END)

    def run_backend(self):
        try:
            monthly = float(self.monthly_var.get())
            years = int(self.years_var.get())
            goal = self.goal_var.get()
        except ValueError:
            messagebox.showerror("Invalid input", "Please enter valid numeric inputs.")
            return

        chosen = [s for s in self.stocks if self.selected[s["ticker"]].get()]
        if len(chosen) < 2:
            messagebox.showerror("Selection error", "Please select at least 2 stocks.")
            return

        input_file = Path("input.csv")
        with open(input_file, "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["monthly_investment", monthly])
            writer.writerow(["years", years])
            writer.writerow(["goal", goal])
            writer.writerow([])
            writer.writerow(["ticker", "price", "growth_rate", "dividend_yield"])
            for stock in chosen:
                writer.writerow([
                    stock["ticker"],
                    stock["price"],
                    stock["growth_rate"],
                    stock["dividend_yield"],
                ])

        try:
            result = subprocess.run(
                [BACKEND_EXE, "input.csv", "results.csv"],
                capture_output=True,
                text=True,
                check=True,
            )
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, "Backend ran successfully.\n\n")
            self.output.insert(tk.END, result.stdout + "\n")

            if RESULTS_FILE.exists():
                self.output.insert(tk.END, "\n--- results.csv ---\n")
                self.output.insert(tk.END, RESULTS_FILE.read_text())

        except subprocess.CalledProcessError as e:
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, "Backend failed.\n\n")
            self.output.insert(tk.END, e.stdout + "\n" + e.stderr)


if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioApp(root)
    root.mainloop()