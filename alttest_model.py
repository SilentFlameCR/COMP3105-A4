"""
Test script for A4codes.py
"""

from altA4codes import learn, compute_accuracy
import os

if __name__ == '__main__':
    # Paths to data folders
    base_path = "/Users/abaan/Downloads/COMP3105-A4"
    path_to_in_domain_train = os.path.join(base_path, "in-domain-train")
    path_to_out_domain_train = os.path.join(base_path, "out-domain-train")
    path_to_in_domain_eval = os.path.join(base_path, "in-domain-eval")
    path_to_out_domain_eval = os.path.join(base_path, "out-domain-eval")

    print("Training model...")
    model = learn(path_to_in_domain_train, path_to_out_domain_train)

    print("\nEvaluating on in-domain data...")
    in_domain_accuracy = compute_accuracy(path_to_in_domain_eval, model)
    print(f"In-domain accuracy: {in_domain_accuracy:.4f}")

    print("\nEvaluating on out-domain data...")
    out_domain_accuracy = compute_accuracy(path_to_out_domain_eval, model)
    print(f"Out-domain accuracy: {out_domain_accuracy:.4f}")

    print(f"\nTarget: In-domain > 0.8, Out-domain ~ 0.1 (random guessing)")
    print(f"Results: In-domain = {in_domain_accuracy:.4f}, Out-domain = {out_domain_accuracy:.4f}")

