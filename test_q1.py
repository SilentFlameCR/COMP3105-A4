"""Test Q1: learn() and compute_accuracy()"""

from A4codes import learn, compute_accuracy

print("=" * 60)
print("Q1 Test: Training classifier on in-domain data")
print("=" * 60)

print("\n[1] Training model on in-domain data...")
print("    (This may take several minutes, especially on CPU)")
model = learn('in-domain-train', 'out-domain-train')
print("    ✓ Training complete.")
print(f"    - Classes: {list(model['class_to_idx'].keys())}")
print(f"    - Device: {model['device']}")

print("\n[2] Evaluating on in-domain data...")
in_acc = compute_accuracy('in-domain-eval', model)
print(f"    ✓ In-domain accuracy: {in_acc:.4f} ({in_acc*100:.2f}%)")

print("\n[3] Evaluating on out-domain data...")
out_acc = compute_accuracy('out-domain-eval', model)
print(f"    ✓ Out-domain accuracy: {out_acc:.4f} ({out_acc*100:.2f}%)")

print("\n" + "=" * 60)
print("RESULTS")
print("=" * 60)
print(f"In-domain accuracy:  {in_acc*100:.2f}%")
print(f"Out-domain accuracy: {out_acc*100:.2f}%")
print("Random baseline:     10.00%")
print()

if in_acc > 0.50:
    print("✓ Good in-domain performance!")
else:
    print("⚠ In-domain accuracy is low — consider more epochs or tuning.")

if out_acc < 0.30:
    print("✓ Out-domain accuracy is low (good for assignment goal!)")
else:
    print("⚠ Out-domain is too high — need stronger domain separation.")

print("=" * 60)
