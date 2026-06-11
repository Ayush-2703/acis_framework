"""
ACIS Command-Line Interface
============================
Usage:
  acis assess  --asset SPS                  Threat assessment for an asset category
  acis attack  --type poisoning --dataset ppe Run a specific attack simulation
  acis matrix                                Print the full risk matrix
  acis list-threats                          List all threat vectors
  acis demo                                  Run a full demo pipeline
"""

from __future__ import annotations

import json
import sys
import time

import click
import numpy as np
from sklearn.ensemble import RandomForestClassifier

BANNER = r"""
   ___   ____ _____  _____
  / _ | / ___/  _/ |/ / _ |
 / __ |/ /___/ //    / __ |
/_/ |_|\___/___/_/|_/_/ |_|

Adversarial Construction Intelligence Security Framework
v1.0.0  |  Yadav et al., ICCCIS-2026
"""

def _get_framework():
    from acis.core.framework import ACISFramework
    return ACISFramework()

def _get_taxonomy():
    from acis.core.threat_taxonomy import ACISThreatTaxonomy
    return ACISThreatTaxonomy()

def _get_matrix():
    from acis.core.risk_matrix import ACISRiskMatrix
    return ACISRiskMatrix()


@click.group()
@click.version_option("1.0.0", prog_name="acis")
def cli():
    """ACIS — Adversarial Construction Intelligence Security Framework."""
    pass


# ---------------------------------------------------------------------------
# assess command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--asset", "-a",
    type=click.Choice(["DIS", "SPS", "AES", "FMA"], case_sensitive=False),
    required=True,
    help="AI asset category to assess.",
)
@click.option("--name",  "-n", default="Unnamed System", help="System name.")
@click.option("--federated",  is_flag=True, help="Uses federated learning.")
@click.option("--physical",   is_flag=True, help="Has physical consequence chain.")
@click.option("--occupant",   is_flag=True, help="Processes occupant data.")
@click.option("--queryable",  is_flag=True, help="Externally queryable via API.")
@click.option("--output", "-o", type=click.Path(), default=None,
              help="Save JSON report to file.")
def assess(asset, name, federated, physical, occupant, queryable, output):
    """Run an ACIS threat assessment for a construction AI system."""
    click.echo(click.style(BANNER, fg="cyan"))

    from acis.core.framework import ACISFramework, SystemProfile
    from acis.core.threat_taxonomy import AssetCategory

    asset_map = {
        "DIS": AssetCategory.DIS, "SPS": AssetCategory.SPS,
        "AES": AssetCategory.AES, "FMA": AssetCategory.FMA,
    }

    profile = SystemProfile(
        name=name,
        asset_category=asset_map[asset.upper()],
        uses_federated_learning=federated,
        has_physical_consequence=physical,
        processes_occupant_data=occupant,
        is_externally_queryable=queryable,
    )

    fw = _get_framework()
    click.echo(click.style(f"  Assessing: {name} [{asset}]", fg="yellow"))
    result = fw.assess_system(profile)
    fw.print_report(result)

    if output:
        fw.export_report(result, path=output)
        click.echo(click.style(f"  ✓ Report saved to {output}", fg="green"))


# ---------------------------------------------------------------------------
# attack command
# ---------------------------------------------------------------------------

@cli.command()
@click.option(
    "--type", "-t", "attack_type",
    type=click.Choice(
        ["poisoning", "targeted-poison", "fgsm", "pgd", "extraction",
         "backdoor", "membership", "federated-poison"],
        case_sensitive=False,
    ),
    required=True,
    help="Attack type to simulate.",
)
@click.option(
    "--dataset", "-d",
    type=click.Choice(["ppe", "bim", "rebar", "benchmark"], case_sensitive=False),
    default="ppe",
    show_default=True,
    help="Dataset to run the attack on.",
)
@click.option("--poison-rate",   default=0.20,  show_default=True, type=float)
@click.option("--epsilon",       default=0.03,  show_default=True, type=float)
@click.option("--n-queries",     default=500,   show_default=True, type=int)
@click.option("--n-steps",       default=40,    show_default=True, type=int)
@click.option("--output", "-o",  type=click.Path(), default=None)
def attack(attack_type, dataset, poison_rate, epsilon, n_queries, n_steps, output):
    """Run an ACIS attack simulation on a construction dataset."""
    click.echo(click.style(BANNER, fg="cyan"))

    from acis.data.datasets import ConstructionBenchmark
    from sklearn.ensemble import RandomForestClassifier

    click.echo(click.style(f"  Loading dataset: {dataset.upper()} ...", fg="yellow"))
    bench = ConstructionBenchmark()
    ds = getattr(bench, f"load_{dataset}" if dataset != "benchmark" else "load_mnist_like")()
    click.echo(f"  {ds}")

    model = RandomForestClassifier(n_estimators=50, random_state=42)
    model.fit(ds.X_train, ds.y_train)
    data  = ds.as_tuple()

    result = None
    click.echo(click.style(f"\n  Running: {attack_type} attack ...\n", fg="magenta"))
    t0 = time.perf_counter()

    if attack_type == "poisoning":
        from acis.attacks.data_poisoning import LabelFlippingAttack
        result = LabelFlippingAttack(poison_rate=poison_rate).run(model, data)

    elif attack_type == "targeted-poison":
        from acis.attacks.data_poisoning import ConstructionPPEPoison
        result = ConstructionPPEPoison(poison_rate=poison_rate).run(model, data)

    elif attack_type in ("fgsm", "pgd"):
        try:
            import torch
            import torch.nn as nn

            class SimpleNet(nn.Module):
                def __init__(self, n_in, n_cls):
                    super().__init__()
                    self.net = nn.Sequential(
                        nn.Linear(n_in, 64), nn.ReLU(),
                        nn.Linear(64, 32),   nn.ReLU(),
                        nn.Linear(32, n_cls)
                    )
                def forward(self, x): return self.net(x)

            n_in  = ds.n_features
            n_cls = ds.n_classes
            net   = SimpleNet(n_in, n_cls)
            X_t   = torch.FloatTensor(ds.X_test[:200])
            y_t   = torch.LongTensor(ds.y_test[:200].astype(int))

            if attack_type == "fgsm":
                from acis.attacks.adversarial_inputs import FGSMAttack
                result = FGSMAttack(epsilon=epsilon).run(net, (X_t, y_t))
            else:
                from acis.attacks.adversarial_inputs import PGDAttack
                result = PGDAttack(epsilon=epsilon, n_steps=n_steps).run(net, (X_t, y_t))
        except ImportError:
            click.echo(click.style(
                "  PyTorch not found. Install: pip install torch", fg="red"
            ))
            sys.exit(1)

    elif attack_type == "extraction":
        from acis.attacks.model_extraction import ModelExtractionAttack
        result = ModelExtractionAttack(n_queries=n_queries).run(model, data)

    elif attack_type == "backdoor":
        from acis.attacks.backdoor_membership import BackdoorAttack
        result = BackdoorAttack().run(model, data)

    elif attack_type == "membership":
        from acis.attacks.backdoor_membership import MembershipInferenceAttack
        result = MembershipInferenceAttack().run(model, data)

    elif attack_type == "federated-poison":
        from acis.federated.federated import FederatedCoordinator
        coord   = FederatedCoordinator(n_rounds=8)
        clients = coord.create_consortium(n_firms=8, n_malicious=2)
        history = coord.train(clients, ds.X_train, ds.y_train, ds.X_test, ds.y_test)
        coord.print_security_report(history)
        elapsed = time.perf_counter() - t0
        click.echo(click.style(f"\n  ✓ Completed in {elapsed:.2f}s", fg="green"))
        return

    if result:
        click.echo(click.style("\n  ── RESULT ──────────────────────────────", fg="cyan"))
        click.echo(result.summary())
        click.echo(click.style("  ────────────────────────────────────────", fg="cyan"))

        if output:
            with open(output, "w") as f:
                json.dump(result.to_dict(), f, indent=2)
            click.echo(click.style(f"\n  ✓ Result saved to {output}", fg="green"))

    elapsed = time.perf_counter() - t0
    click.echo(click.style(f"\n  ✓ Completed in {elapsed:.2f}s", fg="green"))


# ---------------------------------------------------------------------------
# matrix command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--format", "-f", "fmt",
              type=click.Choice(["table", "json", "csv"]),
              default="table", show_default=True)
def matrix(fmt):
    """Print the ACIS risk matrix (Table 2 from the paper)."""
    rm = _get_matrix()

    if fmt == "json":
        click.echo(json.dumps(rm.to_dict(), indent=2))
        return

    if fmt == "csv":
        df = rm.to_dataframe()
        click.echo(df.to_csv())
        return

    # Rich table view
    click.echo(click.style(BANNER, fg="cyan"))
    click.echo(click.style("  ACIS RISK MATRIX  (1=Very Low  5=Critical)\n", bold=True))

    try:
        df = rm.to_dataframe()
        click.echo(df.to_string())
    except ImportError:
        arr = rm.as_numpy()
        headers = ["DATA_POISON", "ADV_INPUT", "EXTRACTION", "INVERSION", "SUPPLY_CHAIN"]
        rows    = ["DIS", "SPS", "AES", "FMA"]
        w = 14
        header_row = " " * 8 + "".join(h.center(w) for h in headers)
        click.echo(header_row)
        click.echo("─" * (8 + w * len(headers)))
        for i, row_name in enumerate(rows):
            vals = "".join(_risk_color(v).center(w) for v in arr[i])
            click.echo(f"  {row_name:<6}{vals}")

    s = rm.summary()
    click.echo(click.style(
        f"\n  Critical cells: {s['critical_cells']} | "
        f"Highest risk asset: {s['highest_risk_asset']} | "
        f"Highest risk threat: {s['highest_risk_threat']}",
        fg="yellow",
    ))


def _risk_color(v: int) -> str:
    labels = {1: "VERY_LOW", 2: "LOW", 3: "MEDIUM", 4: "HIGH", 5: "CRITICAL"}
    return labels.get(v, str(v))


# ---------------------------------------------------------------------------
# list-threats command
# ---------------------------------------------------------------------------

@cli.command("list-threats")
@click.option("--wrong", type=click.Choice(["stealing", "lying", "harming"]),
              default=None)
@click.option("--asset", type=click.Choice(["DIS", "SPS", "AES", "FMA"]),
              default=None)
@click.option("--json", "as_json", is_flag=True)
def list_threats(wrong, asset, as_json):
    """List all ACIS threat vectors, optionally filtered."""
    from acis.core.threat_taxonomy import (
        ACISThreatTaxonomy, WrongType, AssetCategory
    )
    tax = _get_taxonomy()

    if wrong:
        threats = tax.get_threats_by_wrong(WrongType(wrong))
    elif asset:
        asset_map = {"DIS": AssetCategory.DIS, "SPS": AssetCategory.SPS,
                     "AES": AssetCategory.AES, "FMA": AssetCategory.FMA}
        threats = tax.get_threats_for_asset(asset_map[asset])
    else:
        threats = tax.all_threats()

    if as_json:
        click.echo(json.dumps([t.to_dict() for t in threats], indent=2))
        return

    click.echo(click.style(BANNER, fg="cyan"))
    click.echo(click.style(f"  {len(threats)} threat(s) found\n", bold=True))
    for t in threats:
        risk_col = "red" if t.risk_level.value == "CRITICAL" else (
                   "yellow" if t.risk_level.value == "HIGH" else "white")
        click.echo(
            click.style(f"  [{t.risk_level.value:>8}]  ", fg=risk_col) +
            click.style(t.threat_type.value, bold=True) +
            f"  |  sev={t.severity_score}  lik={t.likelihood_score}  "
            f"wrong={t.wrong_type.value}"
        )


# ---------------------------------------------------------------------------
# demo command
# ---------------------------------------------------------------------------

@cli.command()
@click.option("--quick", is_flag=True, help="Run abbreviated demo.")
def demo(quick):
    """Run a full ACIS framework demonstration pipeline."""
    click.echo(click.style(BANNER, fg="cyan"))
    click.echo(click.style("  Running ACIS demo pipeline...\n", bold=True))

    # 1. Framework overview
    from acis.core.framework import ACISFramework, SystemProfile
    from acis.core.threat_taxonomy import AssetCategory
    from acis.data.datasets import ConstructionBenchmark
    from sklearn.ensemble import RandomForestClassifier

    fw = _get_framework()
    click.echo(click.style("  [1/4] Framework initialised", fg="green"))
    click.echo(f"       {fw}")

    # 2. Assess a site perception system
    profile = SystemProfile(
        name="PPE Safety Monitor v2",
        asset_category=AssetCategory.SPS,
        uses_federated_learning=True,
        has_physical_consequence=False,
        is_externally_queryable=False,
    )
    result = fw.assess_system(profile)
    click.echo(click.style(
        f"\n  [2/4] Threat assessment: {profile.name}", fg="green"))
    click.echo(f"       Risk level: {result.overall_risk_level.value} "
               f"({result.overall_risk_score:.2f}/5.00)")
    click.echo(f"       Critical exposure: {result.has_critical_exposure()}")

    # 3. Run a poisoning attack
    click.echo(click.style("\n  [3/4] Data poisoning simulation (PPE dataset)", fg="green"))
    bench  = ConstructionBenchmark()
    ds     = bench.load_ppe(n_samples=600 if quick else 1200)
    model  = RandomForestClassifier(n_estimators=30 if quick else 50, random_state=42)
    model.fit(ds.X_train, ds.y_train)

    from acis.attacks.data_poisoning import ConstructionPPEPoison
    atk    = ConstructionPPEPoison(poison_rate=0.30)
    res    = atk.run(model, ds.as_tuple())
    click.echo(f"       ASR:      {res.attack_success_rate * 100:.1f}%")
    click.echo(f"       Acc drop: {res.accuracy_drop * 100:.1f}%")
    click.echo(f"       Success:  {res.success}")

    # 4. Summary
    click.echo(click.style("\n  [4/4] Risk matrix summary", fg="green"))
    rm = _get_matrix()
    s  = rm.summary()
    click.echo(f"       Critical cells   : {s['critical_cells']}/{s['total_cells']}")
    click.echo(f"       Mean risk score  : {s['mean_risk_score']:.2f}/5.00")
    click.echo(f"       Highest risk asset: {s['highest_risk_asset']}")

    click.echo(click.style(
        "\n  ✓ Demo complete. Run `acis --help` for all commands.\n", fg="cyan", bold=True
    ))


def main():
    cli()


if __name__ == "__main__":
    main()
