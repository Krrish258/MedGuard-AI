#!/usr/bin/env python3
"""
MedGuard-AI — Clinical Decision Support CLI
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

Usage:
    python cli/main.py --patient schema/example_patients.json --id PT-001
    python cli/main.py --patient schema/example_patients.json --id PT-002
    python cli/main.py --patient my_patient.json

Options:
    --patient  PATH   Path to JSON file containing patient data
                      (either a single patient dict or a list of patients)
    --id       STR    Patient ID to select when file contains multiple patients
    --json            Also print raw JSON output at the end
    --help            Show this help message
"""

import argparse
import json
import sys
from pathlib import Path

# Make sure project root is on sys.path
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from rich import box
from rich.columns import Columns
from rich.console import Console
from rich.panel import Panel
from rich.rule import Rule
from rich.table import Table
from rich.text import Text
from rich import print as rprint

from src.pipeline import MedGuardPipeline

console = Console()

BANNER = """
[bold cyan]╔══════════════════════════════════════════════════════════════╗[/bold cyan]
[bold cyan]║[/bold cyan]   [bold white]MedGuard-AI[/bold white]  [dim]Agentic Clinical Decision-Support Framework[/dim]   [bold cyan]║[/bold cyan]
[bold cyan]║[/bold cyan]   [dim]Generative Reasoning  ×  Causal Verification Engine[/dim]        [bold cyan]║[/bold cyan]
[bold cyan]╚══════════════════════════════════════════════════════════════╝[/bold cyan]
[bold red]⚠  RESEARCH PROTOTYPE — NOT FOR CLINICAL USE[/bold red]
"""

DECISION_STYLE = {
    "approved": ("✅  APPROVED",  "bold green"),
    "warning":  ("⚠️   WARNING",  "bold yellow"),
    "abstain":  ("🔴  ABSTAIN",   "bold red"),
}


# ── Rendering helpers ─────────────────────────────────────────────────────────

def render_patient_header(result: dict) -> Panel:
    text  = Text()
    text.append(f"Patient ID : ", style="bold")
    text.append(f"{result['patient_id']}\n")
    text.append(f"Age        : ", style="bold")
    text.append(f"{result['age']} ({result['age_risk']})\n")
    return Panel(text, title="[bold]Patient Profile[/bold]", border_style="cyan", padding=(0, 2))


def render_reasoning_trace(trace: list[dict]) -> Panel:
    table = Table(show_header=True, header_style="bold cyan",
                  box=box.SIMPLE_HEAD, expand=True)
    table.add_column("Step",       style="dim",        width=4, justify="right")
    table.add_column("Action",     style="bold yellow", width=24)
    table.add_column("Thought",    style="white",       ratio=2)
    table.add_column("Observation",style="dim white",   ratio=3)

    for step in trace:
        table.add_row(
            str(step["step"]),
            step["action"],
            step["thought"][:120],
            step["observation"][:160],
        )

    return Panel(table, title="[bold]Layer 1 — Reasoning Trace[/bold]",
                 border_style="blue", padding=(0, 1))


def render_diagnosis(result: dict) -> Panel:
    conf_pct  = f"{result['confidence'] * 100:.1f}%"
    conf_color = (
        "green" if result["confidence"] >= 0.85 else
        "yellow" if result["confidence"] >= 0.60 else
        "red"
    )

    text = Text()
    text.append("Diagnosis       : ", style="bold")
    text.append(f"{result['diagnosis'].title()}\n", style="bold white")
    text.append("Confidence      : ", style="bold")
    text.append(f"{conf_pct}\n", style=f"bold {conf_color}")
    text.append("Drug Class      : ", style="bold")
    text.append(f"{result['drug_class']}\n")
    text.append("Evidence Level  : ", style="bold")
    text.append(f"{result['evidence_level']}\n")

    # Differential
    text.append("\nDifferential Diagnosis:\n", style="bold dim")
    for i, d in enumerate(result["differential"][:5], 1):
        prob = f"{d['probability'] * 100:.1f}%"
        style = "white" if i == 1 else "dim"
        marker = "→" if i == 1 else " "
        text.append(f"  {marker} {i}. {d['disease'].title():<40} {prob}\n", style=style)

    return Panel(text, title="[bold]Diagnosis[/bold]", border_style="magenta", padding=(0, 2))


def render_treatment(result: dict) -> Panel:
    text = Text()
    text.append("Treatment   : ", style="bold")
    text.append(f"{result['treatment']}\n", style="bold white")
    text.append("Drug        : ", style="bold")
    text.append(f"{result['drug_name']}\n")
    text.append("Dose        : ", style="bold")
    text.append(f"{result['dose']}\n")

    return Panel(text, title="[bold]Treatment Plan[/bold]", border_style="magenta", padding=(0, 2))


def render_verification(result: dict) -> Panel:
    safety   = result["safety"]
    score    = safety["score"]
    penalties= safety["penalties"]

    # Score bar
    filled = int(score / 5)
    bar    = "█" * filled + "░" * (20 - filled)
    bar_color = "green" if score >= 85 else "yellow" if score >= 50 else "red"

    text = Text()
    text.append(f"Safety Score : ", style="bold")
    text.append(f"{score}/100  ", style=f"bold {bar_color}")
    text.append(bar + "\n", style=bar_color)
    text.append(f"Risk Level   : {safety['risk_label']}\n", style="bold")

    # Breakdown table
    ptable = Table(show_header=True, header_style="bold dim", box=box.SIMPLE_HEAD)
    ptable.add_column("Check")
    ptable.add_column("Penalty", justify="right")
    ptable.add_column("Status")

    checks = [
        ("Allergy Conflicts",         penalties["allergy"],     safety["allergy_issues"]),
        ("Drug Interactions",         penalties["interaction"],  safety["interaction_issues"]),
        ("History Contraindications", penalties["history"],      safety["history_issues"]),
        ("Outcome Risk Assessment",   penalties["risk"],         []),
    ]
    for name, pen, issues in checks:
        status = "✅ Clear" if pen == 0 else f"❌ -{pen}pts ({len(issues)} issue{'s' if len(issues)!=1 else ''})"
        ptable.add_row(name, str(pen), status)

    ptable.add_row("[bold]TOTAL PENALTY[/bold]", f"[bold]{penalties['total']}[/bold]", "")

    # Issue details
    all_issues = (
        safety["allergy_issues"] +
        safety["interaction_issues"] +
        safety["history_issues"]
    )

    issue_text = Text()
    if all_issues:
        issue_text.append("\nIssue Details:\n", style="bold dim")
        for issue in all_issues:
            sev   = issue.get("severity", "?")
            sev_style = "bold red" if sev == "CRITICAL" else "yellow" if sev == "MAJOR" else "dim"
            msg   = issue.get("message", issue.get("effect", str(issue)))
            issue_text.append(f"  [{sev}] ", style=sev_style)
            issue_text.append(f"{msg[:120]}\n", style="white")

    if safety["risk_factors"]:
        issue_text.append("\nRisk Factors:\n", style="bold dim")
        for rf in safety["risk_factors"]:
            issue_text.append(f"  • {rf}\n", style="dim")

    content = Text.assemble(text, "\n", issue_text)

    from rich.console import Group
    return Panel(
        Group(text, ptable, issue_text),
        title="[bold]Layer 2 — Causal Verification Engine[/bold]",
        border_style="yellow",
        padding=(0, 1),
    )


def render_decision(result: dict) -> Panel:
    decision = result["decision"]
    label, style = DECISION_STYLE.get(decision, (decision.upper(), "bold white"))
    score   = result["safety"]["score"]
    conf    = result["confidence"]

    text = Text()
    text.append(f"\n  {label}\n\n", style=style)
    text.append(f"  {result['message']}\n\n", style="white")
    text.append(f"  Safety Score  : {score}/100\n", style="dim")
    text.append(f"  Confidence    : {conf*100:.1f}%\n", style="dim")

    border = {"approved": "green", "warning": "yellow", "abstain": "red"}.get(decision, "white")
    return Panel(text, title="[bold]Final Decision[/bold]", border_style=border, padding=(0, 2))


def render_symptom_matching(match_report: list[dict]) -> Panel:
    table = Table(show_header=True, header_style="bold dim", box=box.SIMPLE_HEAD)
    table.add_column("Raw Input",     style="white")
    table.add_column("Matched",       style="cyan")
    table.add_column("Score",         justify="right")
    table.add_column("Method",        style="dim")

    for row in match_report:
        matched = row["matched"] or "[dim]Unmatched[/dim]"
        score   = str(row["score"])
        table.add_row(row["raw"], matched, score, row["method"])

    return Panel(table, title="[bold]Symptom Matching Report[/bold]",
                 border_style="cyan", padding=(0, 1))


# ── Main ──────────────────────────────────────────────────────────────────────

def load_patient(path: Path, patient_id: str | None) -> dict:
    with open(path) as f:
        data = json.load(f)

    if isinstance(data, list):
        if not patient_id:
            console.print("[bold yellow]Multiple patients found. Using first patient.[/bold yellow]")
            console.print(f"[dim]Use --id <PATIENT_ID> to select a specific patient.[/dim]")
            return data[0]
        for p in data:
            if str(p.get("patient_id", "")).strip() == patient_id.strip():
                return p
        console.print(f"[bold red]Patient ID '{patient_id}' not found in file.[/bold red]")
        console.print(f"Available IDs: {[p.get('patient_id') for p in data]}")
        sys.exit(1)

    return data


def interactive_input() -> dict:
    from rich.prompt import Prompt, IntPrompt
    console.print("[bold cyan]── Interactive Patient Intake ──[/bold cyan]")
    
    pid = Prompt.ask("Patient ID / Name", default="PT-INTERACTIVE")
    age = IntPrompt.ask("Patient Age")
    
    console.print("[dim]Enter symptoms separated by commas (e.g. 'high fever, dark urine, headache')[/dim]")
    sym_str = Prompt.ask("Symptoms")
    symptoms = [s.strip() for s in sym_str.split(",") if s.strip()]
    if not symptoms:
        console.print("[bold red]At least one symptom is required.[/bold red]")
        sys.exit(1)
    
    hist_str = Prompt.ask("Medical History (comma-separated, optional)", default="")
    history = [s.strip() for s in hist_str.split(",") if s.strip()]
    
    meds_str = Prompt.ask("Current Medications (comma-separated, optional)", default="")
    meds = []
    for m in meds_str.split(","):
        if m.strip():
            meds.append({"name": m.strip(), "dose": "Standard", "frequency": "Daily"})
            
    all_str = Prompt.ask("Allergies (comma-separated, optional)", default="")
    allergies = [s.strip() for s in all_str.split(",") if s.strip()]
    
    console.print("\n[bold green]Intake complete. Passing to MedGuard-AI engine...[/bold green]\n")
    
    return {
        "patient_id": pid,
        "age": age,
        "symptoms": symptoms,
        "medical_history": history,
        "current_medications": meds,
        "allergies": allergies
    }


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="medguard",
        description="MedGuard-AI — Agentic Clinical Decision Support",
    )
    parser.add_argument("--patient",     required=False, help="Path to patient JSON file")
    parser.add_argument("--interactive", action="store_true", help="Launch interactive intake form")
    parser.add_argument("--id",          default=None,  help="Patient ID (if file contains multiple)")
    parser.add_argument("--json",        action="store_true", help="Also print raw JSON output")
    args = parser.parse_args()

    # Header
    console.print(BANNER)

    if args.interactive:
        patient_data = interactive_input()
        console.print(f"[dim]Processing patient: {patient_data.get('patient_id', '?')}...[/dim]\n")
    elif args.patient:
        patient_path = Path(args.patient)
        if not patient_path.exists():
            console.print(f"[bold red]File not found: {patient_path}[/bold red]")
            sys.exit(1)
        patient_data = load_patient(patient_path, args.id)
        console.print(f"[dim]Processing patient: {patient_data.get('patient_id', '?')}...[/dim]\n")
    else:
        parser.print_help()
        sys.exit(1)

    # Run pipeline
    try:
        pipeline = MedGuardPipeline()
        result   = pipeline.run(patient_data)
    except FileNotFoundError as e:
        console.print(f"\n[bold red]Setup error:[/bold red] {e}")
        console.print("\n[bold]Run the setup steps first:[/bold]")
        console.print("  1. python scripts/download_dataset.py")
        console.print("  2. python data/preprocessor.py")
        console.print("  3. python models/train_classifier.py")
        console.print("  4. python knowledge_base/build_knowledge_graph.py")
        console.print("  5. python models/embed_guidelines.py  [optional — PubMedBERT]")
        sys.exit(1)
    except Exception as e:
        console.print(f"\n[bold red]Pipeline error:[/bold red] {e}")
        raise

    # ── Render output ─────────────────────────────────────────────────────────
    console.print(render_patient_header(result))
    console.print()
    console.print(render_reasoning_trace(result["reasoning_trace"]))
    console.print()
    console.print(render_symptom_matching(result["symptom_matching"]))
    console.print()
    console.print(render_diagnosis(result))
    console.print(render_treatment(result))
    console.print()
    console.print(render_verification(result))
    console.print()
    console.print(render_decision(result))
    console.print()
    console.print(Rule(style="dim"))
    console.print("[dim]MedGuard-AI — Research prototype. Not for clinical use.[/dim]")

    # Optional raw JSON
    if args.json:
        console.print("\n[bold dim]── Raw JSON Output ──[/bold dim]")
        console.print_json(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
