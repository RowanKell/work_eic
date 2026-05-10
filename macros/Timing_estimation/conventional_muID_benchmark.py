"""
Conventional muID benchmark: layer-depth based muon/pion separation.

Extracts the conventional muID logic from runTestsAndObjectiveCalc.py
and runs it standalone with ROC curve plotting.

Usage:
  # With existing root files:
  python conventional_muID_benchmark.py --mu_file <mu-.root> --pi_file <pi-.root>

  # With multiple momentum points (provide comma-separated file pairs):
  python conventional_muID_benchmark.py \
      --mu_files file1.root,file2.root \
      --pi_files file1.root,file2.root \
      --p_labels "1 GeV,5 GeV"

  # Generate new simulation data first (requires eic-shell environment):
  python conventional_muID_benchmark.py --generate \
      --momenta 1,2,5,10 --n_events 500
"""

import os
import sys
import argparse
import numpy as np
import awkward as ak
import uproot
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, roc_auc_score


class ConventionalMuID:
    """Layer-depth based muon identification.

    Determines how deep a particle penetrates into the KLM detector
    (number of scintillator layers hit) and uses this as a discriminant
    for muon vs pion separation.
    """

    def __init__(self, superlayer_count=14, steel_thick=55.5, sens_sublayer_thick=20.0,
                 inner_radius=1770, air_gap_thick=0.3, sector_count=8,
                 barrel_length=1500, barrel_offset=18):
        self.superlayer_count = superlayer_count
        self.steel_thick = steel_thick
        self.sens_sublayer_thick = sens_sublayer_thick
        self.inner_radius = inner_radius
        self.air_gap_thick = air_gap_thick
        self.sector_count = sector_count
        self.barrel_length = barrel_length
        self.barrel_offset = barrel_offset

        self.superlayer_dist = (self.steel_thick + self.sens_sublayer_thick * 2
                                + self.air_gap_thick * 4)
        self.outer_radius = self.inner_radius + self.superlayer_count * self.superlayer_dist

        self.first_sens_sublayer_pos = (self.inner_radius + self.steel_thick
                                        + self.air_gap_thick)
        self.adj_sens_sublayer_dist = (self.sens_sublayer_thick
                                       + self.air_gap_thick * 2)

        # Array of start positions for each sensitive sublayer
        self.layer_pos = np.zeros(self.superlayer_count * 2)
        self.layer_pos[::2] = [self.first_sens_sublayer_pos + self.superlayer_dist * i
                               for i in range(self.superlayer_count)]
        self.layer_pos[1::2] = self.layer_pos[::2] + self.adj_sens_sublayer_dist

        # Theta range for particle gun
        self.theta_min = (90 + np.rad2deg(np.arctan(
            (-self.barrel_length / 2 + self.barrel_offset) / self.outer_radius)) * 0.85)
        self.theta_max = (90 + np.rad2deg(np.arctan(
            (self.barrel_length / 2 + self.barrel_offset) / self.outer_radius)) * 0.85)

    def sector_proj_dist(self, xpos, ypos):
        """Project hit position onto nearest sector direction."""
        sector_angle = ((np.arctan2(ypos, xpos) + np.pi / self.sector_count)
                        // (2 * np.pi / self.sector_count) * 2 * np.pi / self.sector_count)
        return xpos * np.cos(sector_angle) + ypos * np.sin(sector_angle)

    def layer_num(self, xpos, ypos):
        """Return the layer number for each hit based on x,y position."""
        pos = self.sector_proj_dist(xpos, ypos)

        within_layer_region = np.logical_and(
            pos * 1.0001 > self.layer_pos[0],
            pos / 1.0001 < self.layer_pos[-1] + self.sens_sublayer_thick)

        superlayer_index = np.where(
            within_layer_region,
            ak.values_astype((pos * 1.0001 - self.layer_pos[0]) // self.superlayer_dist, 'int64'),
            -1)

        layer_pos_dup = ak.Array(np.broadcast_to(
            self.layer_pos,
            (int(ak.num(superlayer_index, axis=0)), len(self.layer_pos))))

        dis_from_first_sublayer = np.where(
            within_layer_region,
            pos - layer_pos_dup[superlayer_index * 2],
            -1)

        in_first_layer = np.logical_and(
            within_layer_region,
            dis_from_first_sublayer / 1.0001 <= self.sens_sublayer_thick)

        in_second_layer = np.logical_and(
            within_layer_region,
            np.logical_and(
                dis_from_first_sublayer * 1.0001 >= self.adj_sens_sublayer_dist,
                dis_from_first_sublayer / 1.0001 <= self.adj_sens_sublayer_dist + self.sens_sublayer_thick))

        hit_layer = np.where(in_first_layer, superlayer_index * 2 + 1, -1)
        hit_layer = np.where(in_second_layer, superlayer_index * 2 + 2, hit_layer)
        return hit_layer

    def pixel_num(self, energy_dep, zpos):
        """Estimate number of detected pixels from energy deposition."""
        inverse = lambda x: 4.9498 / (29.9733 - x + self.barrel_length / 2) - 0.0016796
        efficiency = (inverse(zpos - self.barrel_offset)
                      + inverse(self.barrel_offset - zpos))
        return 10 * energy_dep * (1000 * 1000) * efficiency

    def layer_calc(self, xpos, ypos, zpos, energy_dep):
        """Calculate layers traveled for each particle track.

        Returns:
            layers_traveled: 1D array of max layer reached per track (filtered to >=1)
            layer_counts: counts of tracks terminating at each layer
        """
        hit_layer = self.layer_num(xpos, ypos)
        hit_layer_filtered = np.where(
            self.pixel_num(energy_dep, zpos) >= 2, hit_layer, -2)
        layers_traveled = ak.fill_none(ak.max(hit_layer_filtered, axis=1), -3)
        layer_counts = np.asarray(ak.sum(
            layers_traveled[:, None] == np.arange(1, self.superlayer_count * 2 + 1),
            axis=0))
        return np.asarray(layers_traveled)[layers_traveled >= 1], layer_counts

    def read_hits(self, root_file):
        """Read hit positions and energy from a ROOT file."""
        with uproot.open(root_file) as f:
            hit_x = f['events/HcalBarrelHits.position.x'].array()
            hit_y = f['events/HcalBarrelHits.position.y'].array()
            hit_z = f['events/HcalBarrelHits.position.z'].array()
            # edm4hep renamed EDep -> energy in newer versions
            try:
                hit_edep = f['events/HcalBarrelHits.energy'].array()
            except uproot.exceptions.KeyInFileError:
                hit_edep = f['events/HcalBarrelHits.EDep'].array()
        return hit_x, hit_y, hit_z, hit_edep

    def calc_roc(self, mu_file, pi_file):
        """Calculate ROC curve data for mu/pi separation.

        Returns:
            fpr, tpr, thresholds, auc_score, mu_layers, pi_layers, mu_layer_counts, pi_layer_counts
        """
        mu_x, mu_y, mu_z, mu_edep = self.read_hits(mu_file)
        pi_x, pi_y, pi_z, pi_edep = self.read_hits(pi_file)

        mu_layers_traveled, mu_layer_counts = self.layer_calc(mu_x, mu_y, mu_z, mu_edep)
        pi_layers_traveled, pi_layer_counts = self.layer_calc(pi_x, pi_y, pi_z, pi_edep)

        layers_traveled_tot = np.concatenate((mu_layers_traveled, pi_layers_traveled))
        pid_actual = np.concatenate((
            np.ones_like(mu_layers_traveled),
            np.zeros_like(pi_layers_traveled)))

        # Probability that a particle stopping at this layer is a muon
        pid_layer_prob = np.divide(
            mu_layer_counts,
            mu_layer_counts + pi_layer_counts,
            out=np.zeros(mu_layer_counts.size),
            where=(mu_layer_counts + pi_layer_counts) != 0)

        pid_model = pid_layer_prob[layers_traveled_tot - 1]

        fpr, tpr, thresholds = roc_curve(pid_actual, pid_model)
        auc_score = roc_auc_score(pid_actual, pid_model)

        return (fpr, tpr, thresholds, auc_score,
                mu_layers_traveled, pi_layers_traveled,
                mu_layer_counts, pi_layer_counts)


def plot_roc_curves(results, labels, output_path):
    """Plot ROC curves for one or more momentum points."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # ROC curve
    ax = axes[0]
    for (fpr, tpr, _, auc, _, _, _, _), label in zip(results, labels):
        ax.plot(fpr, tpr, label=f'{label} (AUC = {auc:.3f})', linewidth=2)
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.3, label='Random')
    ax.set_xlabel('False Positive Rate (pion misID)')
    ax.set_ylabel('True Positive Rate (muon efficiency)')
    ax.set_title('Conventional muID: ROC Curve')
    ax.legend(loc='lower right')
    ax.grid(True, alpha=0.3)
    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1])

    # Layer distribution
    ax = axes[1]
    n_layers = results[0][6].size  # mu_layer_counts size
    layer_indices = np.arange(1, n_layers + 1)
    width = 0.35
    for i, ((_, _, _, _, mu_lt, pi_lt, mu_lc, pi_lc), label) in enumerate(zip(results, labels)):
        offset = (i - len(results) / 2 + 0.5) * width * 0.5
        mu_frac = mu_lc / mu_lc.sum() if mu_lc.sum() > 0 else mu_lc
        pi_frac = pi_lc / pi_lc.sum() if pi_lc.sum() > 0 else pi_lc
        ax.step(layer_indices, mu_frac, where='mid', label=f'mu- ({label})',
                linewidth=2, linestyle='-')
        ax.step(layer_indices, pi_frac, where='mid', label=f'pi- ({label})',
                linewidth=2, linestyle='--')
    ax.set_xlabel('Layer Number')
    ax.set_ylabel('Fraction of Tracks')
    ax.set_title('Layer Penetration Depth Distribution')
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Saved plot to {output_path}")


def main():
    parser = argparse.ArgumentParser(description='Conventional muID benchmark')

    # Mode 1: provide existing files
    parser.add_argument('--mu_files', type=str, default=None,
                        help='Comma-separated mu- ROOT files')
    parser.add_argument('--pi_files', type=str, default=None,
                        help='Comma-separated pi- ROOT files')
    parser.add_argument('--p_labels', type=str, default=None,
                        help='Comma-separated momentum labels')

    # Geometry parameters (defaults match standard klmws.xml)
    parser.add_argument('--n_layers', type=int, default=14,
                        help='Number of superlayers')
    parser.add_argument('--steel_thick', type=float, default=55.5,
                        help='Steel thickness in mm')
    parser.add_argument('--scint_thick', type=float, default=20.0,
                        help='Scintillator thickness in mm')

    # Output
    parser.add_argument('--output', type=str, default='conventional_muID_roc.png',
                        help='Output plot filename')

    args = parser.parse_args()

    muid = ConventionalMuID(
        superlayer_count=args.n_layers,
        steel_thick=args.steel_thick,
        sens_sublayer_thick=args.scint_thick)

    print(f"Geometry: {args.n_layers} superlayers, {args.steel_thick}mm steel, "
          f"{args.scint_thick}mm scintillator")
    print(f"Theta range for particle gun: {muid.theta_min:.1f} - {muid.theta_max:.1f} deg")
    print(f"Outer radius: {muid.outer_radius:.1f} mm")

    if args.mu_files and args.pi_files:
        mu_files = args.mu_files.split(',')
        pi_files = args.pi_files.split(',')
        labels = args.p_labels.split(',') if args.p_labels else [f'Set {i}' for i in range(len(mu_files))]

        assert len(mu_files) == len(pi_files), "Must provide same number of mu and pi files"

        results = []
        for mu_f, pi_f, label in zip(mu_files, pi_files, labels):
            print(f"\nProcessing {label}...")
            print(f"  mu- file: {mu_f}")
            print(f"  pi- file: {pi_f}")
            result = muid.calc_roc(mu_f.strip(), pi_f.strip())
            fpr, tpr, thresh, auc, mu_lt, pi_lt, mu_lc, pi_lc = result
            print(f"  mu- tracks with hits in layers: {len(mu_lt)}")
            print(f"  pi- tracks with hits in layers: {len(pi_lt)}")
            print(f"  ROC AUC: {auc:.4f}")
            results.append(result)

        plot_roc_curves(results, labels, args.output)
    else:
        print("\nNo input files provided. Use --mu_files and --pi_files to specify ROOT files.")
        print("Example:")
        print("  python conventional_muID_benchmark.py \\")
        print("    --mu_files scan_mu-_p_1.root,scan_mu-_p_5.root \\")
        print("    --pi_files scan_pi-_p_1.root,scan_pi-_p_5.root \\")
        print("    --p_labels '1 GeV,5 GeV'")
        print("\nOr generate new data with the genMomentumScan.sh script first:")
        print(f"  Recommended theta range: {muid.theta_min:.1f} - {muid.theta_max:.1f} deg")
        sys.exit(1)


if __name__ == '__main__':
    main()
