import sys

# alpha = float(sys.argv[1]) if len(sys.argv) > 1 else 0.5
# l1_ratio = float(sys.argv[2]) if len(sys.argv) > 2 else 0.5

n_estimators =  int(sys.argv[1]) if len(sys.argv)>1  else 200
criterion = sys.argv[2] if len(sys.argv) > 2 and sys.argv[2] in ["gini", "entropy", "log_loss"] else "gini"
min_samples_split = int(sys.argv[3]) if len(sys.argv)>3  else 2

# print(alpha,l1_ratio)

print(n_estimators,criterion,min_samples_split)