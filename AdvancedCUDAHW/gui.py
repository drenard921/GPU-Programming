import csv
import platform
import subprocess
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "stocks.csv"
INPUT_FILE = BASE_DIR / "input.csv"
RESULTS_FILE = BASE_DIR / "results.csv"

if platform.system() == "Windows":
    BACKEND_EXE = BASE_DIR / "solver_app.exe"
else:
    BACKEND_EXE = BASE_DIR / "solver_app"


def load_stocks():
    stocks = []
    with open(DATA_FILE, newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            stocks.append({
                "ticker": row["ticker"],
                "price_now": float(row["price_now"]),
                "cagr_3y": float(row["cagr_3y"]),
                "cagr_5y": float(row["cagr_5y"]),
                "dividend_yield": float(row["dividend_yield"]),
            })
    return stocks


class PortfolioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CUDA Dividend Portfolio Allocation Optimizer")
        self.root.geometry("1150x750")

        self.stocks = load_stocks()
        self.selected = {}
        self.holding_vars = {}

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

        ttk.Label(
            controls, text="Monthly Investment ($):"
        ).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.monthly_var = tk.StringVar(value="500")
        ttk.Entry(
            controls, textvariable=self.monthly_var, width=12
        ).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(
            controls, text="Duration (years):"
        ).grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.years_var = tk.StringVar(value="10")
        ttk.Entry(
            controls, textvariable=self.years_var, width=12
        ).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(
            controls, text="Goal:"
        ).grid(row=0, column=4, sticky="w", padx=5, pady=5)
        self.goal_var = tk.StringVar(value="balanced")
        ttk.Combobox(
            controls,
            textvariable=self.goal_var,
            values=["growth", "income", "balanced"],
            state="readonly",
            width=12,
        ).grid(row=0, column=5, padx=5, pady=5)

        ttk.Label(
            controls, text="Growth Basis:"
        ).grid(row=0, column=6, sticky="w", padx=5, pady=5)
        self.growth_basis_var = tk.StringVar(value="3y")
        ttk.Combobox(
            controls,
            textvariable=self.growth_basis_var,
            values=["3y", "5y"],
            state="readonly",
            width=10,
        ).grid(row=0, column=7, padx=5, pady=5)

        stock_frame = ttk.LabelFrame(
            self.root, text="Select Stocks and Current Holdings", padding=10
        )
        stock_frame.pack(fill="both", expand=False, padx=10, pady=10)

        headers = [
            "Select",
            "Ticker",
            "Price Now",
            "3Y CAGR",
            "5Y CAGR",
            "Dividend Yield",
            "Current Holding ($)",
        ]
        for col, header in enumerate(headers):
            ttk.Label(
                stock_frame, text=header, font=("Arial", 10, "bold")
            ).grid(row=0, column=col, sticky="w", padx=8, pady=4)

        for i, stock in enumerate(self.stocks, start=1):
            ticker = stock["ticker"]

            selected_var = tk.BooleanVar(value=(i <= 4))
            holding_var = tk.StringVar(value="0")

            self.selected[ticker] = selected_var
            self.holding_vars[ticker] = holding_var

            ttk.Checkbutton(
                stock_frame, variable=selected_var
            ).grid(row=i, column=0, padx=8, pady=3, sticky="w")

            ttk.Label(stock_frame, text=ticker).grid(
                row=i, column=1, padx=8, pady=3, sticky="w"
            )
            ttk.Label(stock_frame, text=f'${stock["price_now"]:.2f}').grid(
                row=i, column=2, padx=8, pady=3, sticky="w"
            )
            ttk.Label(stock_frame, text=f'{stock["cagr_3y"] * 100:.2f}%').grid(
                row=i, column=3, padx=8, pady=3, sticky="w"
            )
            ttk.Label(stock_frame, text=f'{stock["cagr_5y"] * 100:.2f}%').grid(
                row=i, column=4, padx=8, pady=3, sticky="w"
            )
            ttk.Label(
                stock_frame, text=f'{stock["dividend_yield"] * 100:.2f}%'
            ).grid(row=i, column=5, padx=8, pady=3, sticky="w")

            ttk.Entry(
                stock_frame, textvariable=holding_var, width=14
            ).grid(row=i, column=6, padx=8, pady=3, sticky="w")

        button_frame = ttk.Frame(self.root, padding=10)
        button_frame.pack(fill="x")

        ttk.Button(
            button_frame,
            text="Run Optimization",
            command=self.run_backend,
        ).pack(side="left", padx=5)

        ttk.Button(
            button_frame,
            text="Clear Output",
            command=self.clear_output,
        ).pack(side="left", padx=5)

        output_frame = ttk.LabelFrame(self.root, text="Results", padding=10)
        output_frame.pack(fill="both", expand=True, padx=10, pady=10)

        self.output = tk.Text(output_frame, wrap="word", font=("Courier New", 10))
        self.output.pack(fill="both", expand=True)

    def clear_output(self):
        self.output.delete("1.0", tk.END)

    def write_input_csv(self, chosen, monthly, years, goal, growth_basis):
        with open(INPUT_FILE, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["monthly_investment", monthly])
            writer.writerow(["years", years])
            writer.writerow(["goal", goal])
            writer.writerow(["growth_basis", growth_basis])
            writer.writerow([])
            writer.writerow([
                "ticker",
                "price_now",
                "cagr_3y",
                "cagr_5y",
                "dividend_yield",
                "current_holding",
            ])

            for stock in chosen:
                writer.writerow([
                    stock["ticker"],
                    stock["price_now"],
                    stock["cagr_3y"],
                    stock["cagr_5y"],
                    stock["dividend_yield"],
                    stock["current_holding"],
                ])

    def run_backend(self):
        try:
            monthly = float(self.monthly_var.get())
            years = int(self.years_var.get())
            goal = self.goal_var.get()
            growth_basis = self.growth_basis_var.get()
        except ValueError:
            messagebox.showerror(
                "Invalid input",
                "Please enter valid numeric inputs.",
            )
            return

        if monthly <= 0 or years <= 0:
            messagebox.showerror(
                "Invalid input",
                "Monthly investment and years must be positive.",
            )
            return

        chosen = []
        for stock in self.stocks:
            ticker = stock["ticker"]
            if self.selected[ticker].get():
                try:
                    current_holding = float(self.holding_vars[ticker].get())
                except ValueError:
                    messagebox.showerror(
                        "Invalid holding",
                        f"Current holding for {ticker} must be numeric.",
                    )
                    return

                if current_holding < 0:
                    messagebox.showerror(
                        "Invalid holding",
                        f"Current holding for {ticker} cannot be negative.",
                    )
                    return

                chosen.append({
                    **stock,
                    "current_holding": current_holding,
                })

        if len(chosen) < 2:
            messagebox.showerror(
                "Selection error",
                "Please select at least 2 stocks.",
            )
            return

        self.write_input_csv(chosen, monthly, years, goal, growth_basis)

        if not BACKEND_EXE.exists():
            messagebox.showerror(
                "Missing backend",
                f"Could not find backend executable:\n{BACKEND_EXE}\n\n"
                "Compile solver.cu first.",
            )
            return

        try:
            result = subprocess.run(
                [str(BACKEND_EXE), str(INPUT_FILE), str(RESULTS_FILE)],
                capture_output=True,
                text=True,
                check=True,
                cwd=BASE_DIR,
            )

            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, "Backend ran successfully.\n\n")
            self.output.insert(tk.END, result.stdout + "\n")

            if RESULTS_FILE.exists():
                self.output.insert(tk.END, "\n--- results.csv ---\n")
                self.output.insert(tk.END, RESULTS_FILE.read_text(encoding="utf-8"))

        except FileNotFoundError:
            self.output.delete("1.0", tk.END)
            self.output.insert(
                tk.END,
                f"Backend executable not found:\n{BACKEND_EXE}\n",
            )
        except OSError as e:
            self.output.delete("1.0", tk.END)
            self.output.insert(
                tk.END,
                f"Backend launch failed:\n{e}\n",
            )
        except subprocess.CalledProcessError as e:
            self.output.delete("1.0", tk.END)
            self.output.insert(tk.END, "Backend failed.\n\n")
            if e.stdout:
                self.output.insert(tk.END, "STDOUT:\n" + e.stdout + "\n")
            if e.stderr:
                self.output.insert(tk.END, "STDERR:\n" + e.stderr + "\n")


if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioApp(root)
    root.mainloop()