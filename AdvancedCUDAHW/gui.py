import csv
import platform
import subprocess
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk, messagebox, filedialog

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "stocks.csv"
INPUT_FILE = BASE_DIR / "input.csv"
RESULTS_FILE = BASE_DIR / "results.csv"
PY_RESULTS_FILE = BASE_DIR / "results_python.csv"

if platform.system() == "Windows":
    BACKEND_EXE = BASE_DIR / "solver_app.exe"
else:
    BACKEND_EXE = BASE_DIR / "solver_app"


ALPHA = 0.25


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


def active_growth_rate(stock, basis):
    return stock["cagr_5y"] if basis == "5y" else stock["cagr_3y"]


def build_score_vector(stocks, goal, growth_basis):
    scores = []
    for stock in stocks:
        growth = active_growth_rate(stock, growth_basis)
        dividend = stock["dividend_yield"]
        if goal == "growth":
            score = growth
        elif goal == "income":
            score = dividend
        else:
            score = 0.5 * growth + 0.5 * dividend
        scores.append(score)

    min_score = min(scores)
    if min_score < 0:
        scores = [score - min_score for score in scores]

    return [score + 1.0e-3 for score in scores]


def solve_linear_system_python(matrix, rhs):
    n = len(rhs)
    a = [row[:] for row in matrix]
    b = rhs[:]

    for pivot in range(n):
        max_row = max(range(pivot, n), key=lambda r: abs(a[r][pivot]))
        if abs(a[max_row][pivot]) < 1.0e-12:
            raise ValueError("Singular linear system in Python solver")
        if max_row != pivot:
            a[pivot], a[max_row] = a[max_row], a[pivot]
            b[pivot], b[max_row] = b[max_row], b[pivot]

        pivot_val = a[pivot][pivot]
        for row in range(pivot + 1, n):
            factor = a[row][pivot] / pivot_val
            for col in range(pivot, n):
                a[row][col] -= factor * a[pivot][col]
            b[row] -= factor * b[pivot]

    x = [0.0] * n
    for row in range(n - 1, -1, -1):
        total = b[row]
        for col in range(row + 1, n):
            total -= a[row][col] * x[col]
        x[row] = total / a[row][row]
    return x


def solve_weights_python(scores):
    n = len(scores)
    matrix = []
    for row in range(n):
        row_vals = []
        for col in range(n):
            val = ALPHA
            if row == col:
                val += 1.0
            row_vals.append(val)
        matrix.append(row_vals)

    weights = solve_linear_system_python(matrix, scores)
    weights = [max(0.0, weight) for weight in weights]
    total = sum(weights)
    if total <= 0:
        return [1.0 / n] * n
    return [weight / total for weight in weights]


def write_python_results_csv(path, invested_totals, cash_totals, annual_dividends):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow([
            "month",
            "invested_value",
            "cash_dividends",
            "total_value",
            "total_dividend_income",
            "monthly_dividend_income",
        ])
        for month, (invested, cash, annual_div) in enumerate(
            zip(invested_totals, cash_totals, annual_dividends)
        ):
            writer.writerow([
                month,
                f"{invested:.2f}",
                f"{cash:.2f}",
                f"{(invested + cash):.2f}",
                f"{annual_div:.2f}",
                f"{(annual_div / 12.0):.2f}",
            ])


def run_python_solver(chosen, monthly, years, goal, growth_basis):
    start = time.perf_counter()

    scores = build_score_vector(chosen, goal, growth_basis)
    weights = solve_weights_python(scores)

    n = len(chosen)
    months = years * 12

    values = [stock["current_holding"] for stock in chosen]
    cash_dividends = [0.0] * n
    annual_dividend_rates = [stock["dividend_yield"] for stock in chosen]
    monthly_growth_rates = [
        active_growth_rate(stock, growth_basis) / 12.0 for stock in chosen
    ]
    monthly_dividend_rates = [stock["dividend_yield"] / 12.0 for stock in chosen]
    contribution_vector = [weight * monthly for weight in weights]
    current_total = sum(values)

    current_weights = [0.0] * n
    if current_total > 0:
        current_weights = [value / current_total for value in values]

    invested_series = [current_total]
    cash_series = [0.0]
    annual_income_series = [
        sum(
            value * annual_rate
            for value, annual_rate in zip(values, annual_dividend_rates)
        )
    ]

    for _ in range(months):
        for i in range(n):
            start_value = values[i]
            dividend_cash = start_value * monthly_dividend_rates[i]
            end_value = (
                start_value * (1.0 + monthly_growth_rates[i])
                + contribution_vector[i]
            )
            if chosen[i]["reinvest_dividends"]:
                end_value += dividend_cash
            else:
                cash_dividends[i] += dividend_cash
            values[i] = end_value

        invested_total = sum(values)
        cash_total = sum(cash_dividends)
        annual_income = sum(
            value * annual_rate
            for value, annual_rate in zip(values, annual_dividend_rates)
        )
        invested_series.append(invested_total)
        cash_series.append(cash_total)
        annual_income_series.append(annual_income)

    elapsed_ms = (time.perf_counter() - start) * 1000.0
    write_python_results_csv(
        PY_RESULTS_FILE,
        invested_series,
        cash_series,
        annual_income_series,
    )

    final_total = invested_series[-1] + cash_series[-1]
    initial_annual_dividend = annual_income_series[0]
    final_annual_dividend = annual_income_series[-1]

    lines = []
    lines.append("Python baseline ran successfully.\n")
    lines.append(f"CPU runtime: {elapsed_ms:.3f} ms\n")
    lines.append(f"Goal: {goal}\n")
    lines.append(f"Growth basis: {growth_basis}\n")
    lines.append(f"Monthly investment: ${monthly:.2f}\n")
    lines.append(f"Duration: {years} years\n")
    lines.append(f"Current portfolio value: ${current_total:.2f}\n\n")
    lines.append("Current vs Recommended allocations:\n")

    for stock, current_weight, weight in zip(chosen, current_weights, weights):
        growth = active_growth_rate(stock, growth_basis)
        reinvest = "yes" if stock["reinvest_dividends"] else "no"
        lines.append(
            f"  {stock['ticker']} | current: {current_weight * 100.0:.2f}%"
            f" | recommended: {weight * 100.0:.2f}%"
            f" | growth: {growth * 100.0:.2f}%"
            f" | dividend: {stock['dividend_yield'] * 100.0:.2f}%"
            f" | reinvest: {reinvest}\n"
        )

    lines.append("\nIncome comparison:\n")
    lines.append(
        f"  Current monthly dividend income: ${initial_annual_dividend / 12.0:.2f}\n"
    )
    lines.append(
        f"  Current annual dividend income:  ${initial_annual_dividend:.2f}\n"
    )
    lines.append(
        f"  Projected monthly dividend income at end: ${final_annual_dividend / 12.0:.2f}\n"
    )
    lines.append(
        f"  Projected annual dividend income at end:  ${final_annual_dividend:.2f}\n"
    )
    lines.append(f"\nProjected final value: ${final_total:.2f}\n")
    lines.append(f"Wrote results to {PY_RESULTS_FILE.name}\n")
    lines.append("\n--- results_python.csv ---\n")
    lines.append(PY_RESULTS_FILE.read_text(encoding="utf-8"))

    return "".join(lines)


class PortfolioApp:
    def __init__(self, root):
        self.root = root
        self.root.title("CUDA Dividend Portfolio Allocation Optimizer")
        self.root.geometry("1450x920")

        self.all_stocks = load_stocks()
        self.stocks = [dict(stock) for stock in self.all_stocks]
        self.selected = {}
        self.holding_vars = {}
        self.reinvest_vars = {}

        self.stock_frame = None
        self.stock_canvas = None
        self.stock_canvas_window = None
        self.stock_vscroll = None
        self.stock_hscroll = None

        self.stock_source_var = tk.StringVar(
            value=f"Built-in dataset: {DATA_FILE.name}"
        )

        self._build_ui()

    def normalize_bool(self, value, default=False):
        if value is None:
            return default
        text = str(value).strip().lower()
        if text in {"1", "true", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "no", "n", "off"}:
            return False
        return default

    def merge_stock_rows(self, rows):
        base_map = {stock["ticker"].upper(): dict(stock) for stock in self.all_stocks}
        merged = []

        for row in rows:
            if not row:
                continue

            clean = {
                str(k).strip().lower(): ("" if v is None else str(v).strip())
                for k, v in row.items()
                if k is not None
            }

            ticker = (
                clean.get("ticker")
                or clean.get("symbol")
                or ""
            ).upper().strip()

            if not ticker:
                continue

            stock = dict(base_map.get(ticker, {"ticker": ticker}))

            aliases = {
                "price_now": ["price_now", "latest", "price", "last"],
                "cagr_3y": ["cagr_3y"],
                "cagr_5y": ["cagr_5y"],
                "dividend_yield": ["dividend_yield", "yield", "div_yield"],
            }

            for target_field, candidates in aliases.items():
                for candidate in candidates:
                    value = clean.get(candidate, "")
                    if value != "":
                        stock[target_field] = float(
                            value.replace("%", "")
                        ) / 100.0 if (
                            target_field in {"cagr_3y", "cagr_5y", "dividend_yield"}
                            and "%" in value
                        ) else float(value)
                        break

            for field in ["price_now", "cagr_3y", "cagr_5y", "dividend_yield"]:
                if field not in stock:
                    raise ValueError(
                        f"CSV row for {ticker} is missing '{field}' and it was not found in {DATA_FILE.name}."
                    )

            stock["selected_default"] = self.normalize_bool(
                clean.get("selected"), default=True
            )
            stock["current_holding_default"] = clean.get("current_holding", "0") or "0"
            stock["reinvest_default"] = self.normalize_bool(
                clean.get("reinvest_dividends"), default=True
            )
            merged.append(stock)

        if len(merged) < 2:
            raise ValueError("Need at least 2 valid stock rows after import.")

        deduped = []
        seen = set()
        for stock in merged:
            ticker = stock["ticker"].upper()
            if ticker in seen:
                continue
            seen.add(ticker)
            deduped.append(stock)

        return deduped

    def load_stock_csv_file(self):
        path = filedialog.askopenfilename(
            title="Select stock picks CSV",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
            initialdir=str(BASE_DIR),
        )
        if not path:
            return

        try:
            with open(path, newline="", encoding="utf-8-sig") as f:
                rows = list(csv.DictReader(f))
            stocks = self.merge_stock_rows(rows)
            self.stocks = stocks
            self.stock_source_var.set(f"Imported file: {Path(path).name}")
            self.render_stock_rows()
            messagebox.showinfo(
                "Import complete",
                f"Loaded {len(self.stocks)} stock rows from {Path(path).name}.",
            )
        except Exception as e:
            messagebox.showerror("CSV import failed", str(e))

    def load_stock_csv_folder(self):
        folder = filedialog.askdirectory(
            title="Select folder containing stock pick CSV files",
            initialdir=str(BASE_DIR),
            mustexist=True,
        )
        if not folder:
            return

        try:
            csv_paths = sorted(Path(folder).glob("*.csv"))
            if not csv_paths:
                raise ValueError("No CSV files were found in the selected folder.")

            rows = []
            for csv_path in csv_paths:
                with open(csv_path, newline="", encoding="utf-8-sig") as f:
                    rows.extend(list(csv.DictReader(f)))

            stocks = self.merge_stock_rows(rows)
            self.stocks = stocks
            self.stock_source_var.set(
                f"Imported folder: {Path(folder).name} ({len(csv_paths)} CSVs)"
            )
            self.render_stock_rows()
            messagebox.showinfo(
                "Folder import complete",
                f"Loaded {len(self.stocks)} unique stock rows from {len(csv_paths)} CSV file(s).",
            )
        except Exception as e:
            messagebox.showerror("Folder import failed", str(e))

    def reset_to_default_stocks(self):
        self.stocks = [dict(stock) for stock in self.all_stocks]
        self.stock_source_var.set(f"Built-in dataset: {DATA_FILE.name}")
        self.render_stock_rows()

    def on_stock_frame_configure(self, _event=None):
        if self.stock_canvas is not None:
            self.stock_canvas.configure(scrollregion=self.stock_canvas.bbox("all"))

    def on_stock_canvas_configure(self, event):
        if self.stock_canvas is not None and self.stock_canvas_window is not None:
            self.stock_canvas.itemconfigure(
                self.stock_canvas_window,
                width=event.width,
            )

    def _bind_mousewheel(self):
        self.stock_canvas.bind_all("<MouseWheel>", self._on_mousewheel_windows)
        self.stock_canvas.bind_all("<Button-4>", self._on_mousewheel_linux_up)
        self.stock_canvas.bind_all("<Button-5>", self._on_mousewheel_linux_down)

    def _unbind_mousewheel(self):
        self.stock_canvas.unbind_all("<MouseWheel>")
        self.stock_canvas.unbind_all("<Button-4>")
        self.stock_canvas.unbind_all("<Button-5>")

    def _on_mousewheel_windows(self, event):
        if self.stock_canvas is not None:
            self.stock_canvas.yview_scroll(int(-1 * (event.delta / 120)), "units")

    def _on_mousewheel_linux_up(self, _event):
        if self.stock_canvas is not None:
            self.stock_canvas.yview_scroll(-1, "units")

    def _on_mousewheel_linux_down(self, _event):
        if self.stock_canvas is not None:
            self.stock_canvas.yview_scroll(1, "units")

    def select_all_stocks(self):
        for var in self.selected.values():
            var.set(True)

    def clear_all_stocks(self):
        for var in self.selected.values():
            var.set(False)

    def render_stock_rows(self):
        self.selected = {}
        self.holding_vars = {}
        self.reinvest_vars = {}

        for child in self.stock_frame.winfo_children():
            child.destroy()

        headers = [
            "Select",
            "Ticker",
            "Price Now",
            "3Y CAGR",
            "5Y CAGR",
            "Dividend Yield",
            "Current Holding ($)",
            "Reinvest Dividends",
        ]

        for col, header in enumerate(headers):
            ttk.Label(
                self.stock_frame,
                text=header,
                font=("Arial", 10, "bold"),
            ).grid(row=0, column=col, sticky="w", padx=8, pady=4)

        for i, stock in enumerate(self.stocks, start=1):
            ticker = stock["ticker"]

            selected_var = tk.BooleanVar(value=stock.get("selected_default", i <= 4))
            holding_var = tk.StringVar(
                value=str(stock.get("current_holding_default", "0"))
            )
            reinvest_var = tk.BooleanVar(value=stock.get("reinvest_default", True))

            self.selected[ticker] = selected_var
            self.holding_vars[ticker] = holding_var
            self.reinvest_vars[ticker] = reinvest_var

            ttk.Checkbutton(
                self.stock_frame,
                variable=selected_var,
            ).grid(row=i, column=0, padx=8, pady=3, sticky="w")

            ttk.Label(self.stock_frame, text=ticker).grid(
                row=i, column=1, padx=8, pady=3, sticky="w"
            )
            ttk.Label(self.stock_frame, text=f'${stock["price_now"]:.2f}').grid(
                row=i, column=2, padx=8, pady=3, sticky="w"
            )
            ttk.Label(self.stock_frame, text=f'{stock["cagr_3y"] * 100:.2f}%').grid(
                row=i, column=3, padx=8, pady=3, sticky="w"
            )
            ttk.Label(self.stock_frame, text=f'{stock["cagr_5y"] * 100:.2f}%').grid(
                row=i, column=4, padx=8, pady=3, sticky="w"
            )
            ttk.Label(
                self.stock_frame,
                text=f'{stock["dividend_yield"] * 100:.2f}%',
            ).grid(row=i, column=5, padx=8, pady=3, sticky="w")

            ttk.Entry(
                self.stock_frame,
                textvariable=holding_var,
                width=14,
            ).grid(row=i, column=6, padx=8, pady=3, sticky="w")

            ttk.Checkbutton(
                self.stock_frame,
                variable=reinvest_var,
            ).grid(row=i, column=7, padx=8, pady=3, sticky="w")

        for col in range(len(headers)):
            self.stock_frame.grid_columnconfigure(col, weight=1)

        self.on_stock_frame_configure()

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
            controls,
            text="Monthly Investment ($):",
        ).grid(row=0, column=0, sticky="w", padx=5, pady=5)
        self.monthly_var = tk.StringVar(value="500")
        ttk.Entry(
            controls,
            textvariable=self.monthly_var,
            width=12,
        ).grid(row=0, column=1, padx=5, pady=5)

        ttk.Label(
            controls,
            text="Duration (years):",
        ).grid(row=0, column=2, sticky="w", padx=5, pady=5)
        self.years_var = tk.StringVar(value="10")
        ttk.Entry(
            controls,
            textvariable=self.years_var,
            width=12,
        ).grid(row=0, column=3, padx=5, pady=5)

        ttk.Label(
            controls,
            text="Goal:",
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
            controls,
            text="Growth Basis:",
        ).grid(row=0, column=6, sticky="w", padx=5, pady=5)
        self.growth_basis_var = tk.StringVar(value="3y")
        ttk.Combobox(
            controls,
            textvariable=self.growth_basis_var,
            values=["3y", "5y"],
            state="readonly",
            width=10,
        ).grid(row=0, column=7, padx=5, pady=5)

        self.run_python_var = tk.BooleanVar(value=False)
        ttk.Checkbutton(
            controls,
            text="Also run Python baseline",
            variable=self.run_python_var,
        ).grid(row=0, column=8, padx=10, pady=5, sticky="w")

        import_frame = ttk.LabelFrame(
            self.root,
            text="Stock Universe Source",
            padding=10,
        )
        import_frame.pack(fill="x", padx=10, pady=10)

        ttk.Label(
            import_frame,
            textvariable=self.stock_source_var,
            font=("Arial", 10, "bold"),
        ).pack(anchor="w", pady=(0, 6))

        import_buttons = ttk.Frame(import_frame)
        import_buttons.pack(fill="x")

        ttk.Button(
            import_buttons,
            text="Load Picks CSV...",
            command=self.load_stock_csv_file,
        ).pack(side="left", padx=5)

        ttk.Button(
            import_buttons,
            text="Load Folder of CSVs...",
            command=self.load_stock_csv_folder,
        ).pack(side="left", padx=5)

        ttk.Button(
            import_buttons,
            text="Reset to Built-in stocks.csv",
            command=self.reset_to_default_stocks,
        ).pack(side="left", padx=5)

        selection_buttons = ttk.Frame(import_frame)
        selection_buttons.pack(fill="x", pady=(8, 0))

        ttk.Button(
            selection_buttons,
            text="Select All",
            command=self.select_all_stocks,
        ).pack(side="left", padx=5)

        ttk.Button(
            selection_buttons,
            text="Clear All",
            command=self.clear_all_stocks,
        ).pack(side="left", padx=5)

        stock_container = ttk.LabelFrame(
            self.root,
            text="Select Stocks and Current Holdings",
            padding=10,
        )
        stock_container.pack(fill="both", expand=False, padx=10, pady=10)

        canvas_holder = ttk.Frame(stock_container)
        canvas_holder.pack(fill="both", expand=True)

        self.stock_canvas = tk.Canvas(canvas_holder, height=320)
        self.stock_vscroll = ttk.Scrollbar(
            canvas_holder,
            orient="vertical",
            command=self.stock_canvas.yview,
        )
        self.stock_hscroll = ttk.Scrollbar(
            stock_container,
            orient="horizontal",
            command=self.stock_canvas.xview,
        )

        self.stock_canvas.configure(
            yscrollcommand=self.stock_vscroll.set,
            xscrollcommand=self.stock_hscroll.set,
        )

        self.stock_vscroll.pack(side="right", fill="y")
        self.stock_canvas.pack(side="left", fill="both", expand=True)
        self.stock_hscroll.pack(fill="x")

        self.stock_frame = ttk.Frame(self.stock_canvas)
        self.stock_canvas_window = self.stock_canvas.create_window(
            (0, 0),
            window=self.stock_frame,
            anchor="nw",
        )

        self.stock_frame.bind("<Configure>", self.on_stock_frame_configure)
        self.stock_canvas.bind("<Configure>", self.on_stock_canvas_configure)

        self.stock_canvas.bind("<Enter>", lambda _e: self._bind_mousewheel())
        self.stock_canvas.bind("<Leave>", lambda _e: self._unbind_mousewheel())

        self.render_stock_rows()

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

        runtime_frame = ttk.LabelFrame(
            self.root,
            text="Runtime Summary",
            padding=10,
        )
        runtime_frame.pack(fill="x", padx=10, pady=5)

        self.runtime_label = ttk.Label(
            runtime_frame,
            text="GPU runtime: not run yet | Python CPU runtime: disabled",
            font=("Arial", 10, "bold"),
        )
        self.runtime_label.pack(anchor="w")

        outputs = ttk.Panedwindow(self.root, orient=tk.HORIZONTAL)
        outputs.pack(fill="both", expand=True, padx=10, pady=10)

        gpu_frame = ttk.LabelFrame(outputs, text="GPU / CUDA Results", padding=10)
        self.gpu_output = tk.Text(gpu_frame, wrap="word", font=("Courier New", 10))
        self.gpu_output.pack(fill="both", expand=True)
        outputs.add(gpu_frame, weight=1)

        py_frame = ttk.LabelFrame(outputs, text="Python Baseline Results", padding=10)
        self.python_output = tk.Text(py_frame, wrap="word", font=("Courier New", 10))
        self.python_output.pack(fill="both", expand=True)
        outputs.add(py_frame, weight=1)

    def clear_output(self):
        self.gpu_output.delete("1.0", tk.END)
        self.python_output.delete("1.0", tk.END)
        self.runtime_label.config(
            text="GPU runtime: not run yet | Python CPU runtime: disabled"
        )

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
                "reinvest_dividends",
            ])

            for stock in chosen:
                writer.writerow([
                    stock["ticker"],
                    stock["price_now"],
                    stock["cagr_3y"],
                    stock["cagr_5y"],
                    stock["dividend_yield"],
                    stock["current_holding"],
                    int(stock["reinvest_dividends"]),
                ])

    def gather_selected_stocks(self):
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
            return None

        if monthly <= 0 or years <= 0:
            messagebox.showerror(
                "Invalid input",
                "Monthly investment and years must be positive.",
            )
            return None

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
                    return None

                if current_holding < 0:
                    messagebox.showerror(
                        "Invalid holding",
                        f"Current holding for {ticker} cannot be negative.",
                    )
                    return None

                chosen.append({
                    **stock,
                    "current_holding": current_holding,
                    "reinvest_dividends": self.reinvest_vars[ticker].get(),
                })

        if len(chosen) < 2:
            messagebox.showerror(
                "Selection error",
                "Please select at least 2 stocks.",
            )
            return None

        return monthly, years, goal, growth_basis, chosen

    def run_backend(self):
        gathered = self.gather_selected_stocks()
        if gathered is None:
            return

        monthly, years, goal, growth_basis, chosen = gathered
        self.write_input_csv(chosen, monthly, years, goal, growth_basis)

        gpu_wall_ms = None
        py_ms = None
        self.gpu_output.delete("1.0", tk.END)
        self.python_output.delete("1.0", tk.END)

        if not BACKEND_EXE.exists():
            messagebox.showerror(
                "Missing backend",
                f"Could not find backend executable:\n{BACKEND_EXE}\n\n"
                "Compile solver.cu first.",
            )
            return

        try:
            gpu_start = time.perf_counter()
            result = subprocess.run(
                [str(BACKEND_EXE), str(INPUT_FILE), str(RESULTS_FILE)],
                capture_output=True,
                text=True,
                check=True,
                cwd=BASE_DIR,
            )
            gpu_wall_ms = (time.perf_counter() - gpu_start) * 1000.0

            self.gpu_output.insert(tk.END, "CUDA backend ran successfully.\n\n")
            self.gpu_output.insert(
                tk.END,
                f"GPU wall runtime from GUI: {gpu_wall_ms:.3f} ms\n\n",
            )
            self.gpu_output.insert(tk.END, result.stdout + "\n")

            if RESULTS_FILE.exists():
                self.gpu_output.insert(tk.END, "\n--- results.csv ---\n")
                self.gpu_output.insert(
                    tk.END,
                    RESULTS_FILE.read_text(encoding="utf-8"),
                )

        except FileNotFoundError:
            self.gpu_output.insert(
                tk.END,
                f"Backend executable not found:\n{BACKEND_EXE}\n",
            )
            return
        except OSError as e:
            self.gpu_output.insert(
                tk.END,
                f"Backend launch failed:\n{e}\n",
            )
            return
        except subprocess.CalledProcessError as e:
            self.gpu_output.insert(tk.END, "Backend failed.\n\n")
            if e.stdout:
                self.gpu_output.insert(tk.END, "STDOUT:\n" + e.stdout + "\n")
            if e.stderr:
                self.gpu_output.insert(tk.END, "STDERR:\n" + e.stderr + "\n")
            return

        if self.run_python_var.get():
            py_start = time.perf_counter()
            python_report = run_python_solver(
                chosen,
                monthly,
                years,
                goal,
                growth_basis,
            )
            py_ms = (time.perf_counter() - py_start) * 1000.0
            self.python_output.insert(tk.END, python_report)
        else:
            self.python_output.insert(
                tk.END,
                "Python baseline is disabled. Enable "
                "'Also run Python baseline' to compare CPU and GPU runtimes.\n",
            )

        gpu_text = (
            f"GPU runtime: {gpu_wall_ms:.3f} ms"
            if gpu_wall_ms is not None else "GPU runtime: failed"
        )
        py_text = (
            f"Python CPU runtime: {py_ms:.3f} ms"
            if py_ms is not None else "Python CPU runtime: disabled"
        )
        self.runtime_label.config(text=f"{gpu_text} | {py_text}")


if __name__ == "__main__":
    root = tk.Tk()
    app = PortfolioApp(root)
    root.mainloop()