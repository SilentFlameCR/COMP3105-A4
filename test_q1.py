"""Test Q1: learn() and compute_accuracy()"""
from A4codes import learn, compute_accuracy

print("="*60)
print("Q1 Test: Training classifier on in-domain data")
print("="*60)

# Step 1: Train model
print("\n[1] Training model on in-domain data...")
print("    (This may take a few minutes)")
model = learn('in-domain-train', 'out-domain-train')
print(f"    ✓ Training complete!")
print(f"    - Classes: {model['class_names']}")
print(f"    - Device: {model['device']}")

# Step 2: Evaluate on in-domain
print("\n[2] Evaluating on in-domain data...")
in_acc = compute_accuracy('in-domain-eval', model)
print(f"    ✓ In-domain accuracy: {in_acc:.4f} ({in_acc*100:.2f}%)")

# Step 3: Evaluate on out-domain
print("\n[3] Evaluating on out-domain data...")
out_acc = compute_accuracy('out-domain-eval', model)
print(f"    ✓ Out-domain accuracy: {out_acc:.4f} ({out_acc*100:.2f}%)")

# Analysis
print("\n" + "="*60)
print("RESULTS")
print("="*60)
print(f"In-domain:  {in_acc*100:.2f}%")
print(f"Out-domain: {out_acc*100:.2f}%")
print(f"Random:     10.00% (baseline for 10 classes)")
print()

if in_acc > 0.6:
    print("✓ Good in-domain performance!")
else:
    print("⚠ In-domain needs improvement")

if out_acc < 0.3:
    print("✓ Out-domain is poor (good for our goal!)")
else:
    print("⚠ Out-domain too high - need to reduce generalization")

print("="*60)
