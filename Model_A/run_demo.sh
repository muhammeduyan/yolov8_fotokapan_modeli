#!/bin/zsh
set -e
cd "$(dirname "$0")"

PYTHON_BIN=".venv/bin/python"
if [ ! -x "$PYTHON_BIN" ]; then
    PYTHON_BIN="python3"
fi

echo "=============================================================================="
echo "🚀 MODEL A (LOCAL WILD BOAR v0.4) - KNN & OOD FİLTRE DEMOSU"
echo "=============================================================================="
echo ""
echo "👉 1. ADIM: Dağılım Dışı (OOD) Yaban Keçisi Görseli Test Ediliyor..."
echo "Komut: $PYTHON_BIN src/predict.py --source samples/02_same_goat.jpg"
echo ""
"$PYTHON_BIN" src/predict.py --source samples/02_same_goat.jpg

echo ""
echo "👉 2. ADIM: Test Setinden Gerçek Yaban Domuzu Görseli Test Ediliyor..."
echo "Komut: $PYTHON_BIN src/predict.py --source samples/01_wild_boar.jpg"
echo ""
"$PYTHON_BIN" src/predict.py --source samples/01_wild_boar.jpg

echo ""
echo "👉 3. ADIM: Kalibrasyon Raporu & 100 Negatif Görsel Benchmark..."
echo "Komut: $PYTHON_BIN src/predict.py --benchmark"
echo ""
"$PYTHON_BIN" src/predict.py --benchmark
