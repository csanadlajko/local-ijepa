REPO="csanadl/local-ijepa-geometry"
MODEL_FOLDERS=("results/euclidean" "results/hyperbolic")

for model in "${MODEL_FOLDERS[@]}"; do
    hf upload $REPO $model
done
