# Towards a Perspectivist Turn in Argument Quality Assessment

## Database

## Experiments
To reproduce our analyses, request the raw data from the respective original authors.

Preprocess the raw data (see [expected_format.md](expected_format.md) for details on the expected data).

To do so, install the requirements
```bash
python3 -m pip install -r requirements.txt
```

and run the preprocessing scripts:
```bash
python3 preprocess_dagstuhl.py --path path/to/expert/csv --output output/directory --annotation_type expert
python3 preprocess_dagstuhl.py --path path/to/crowd/csv --output output/directory --annotation_type crowd
python3 preprocess_dagstuhl.py --path path/to/novice/directory --output output/directory --annotation_type novice
python3 preprocess_gaq.py --path path/to/GAQ/directory --ouput output/directory
```
The analyses done on the preprocessed data can be found in [analysis.ipynb](analysis.ipynb).

## Reference
```bibtex
TODO
```